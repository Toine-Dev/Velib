import os
from data.preprocessing import coerce_velib_types, standardize_columns
from utils.config import csv_url, database_url, table_name
from data.ingestion import _download_to_tempfile
from sqlalchemy import create_engine, text
import sys
import time
from sqlalchemy.exc import OperationalError
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from sqlalchemy import text

from utils.utils import insert_on_conflict_do_nothing



# ==============================
# Create table if not exists with appropriate schema, indexes and unique constraints to prevent duplicates and optimize queries.
# ==============================

def ensure_velib_raw_schema(engine):
    print("Ensuring velib_raw table schema exists with correct types and indexes...", flush=True)
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS velib_raw (
                identifiant_du_site_de_comptage BIGINT,
                nom_du_site_de_comptage TEXT,
                comptage_horaire BIGINT,
                date_et_heure_de_comptage TIMESTAMP,
                coordonnees_geographiques TEXT
            );
        """))

        # Index for time filtering / max(date) queries
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_velib_raw_date
            ON velib_raw (date_et_heure_de_comptage);
        """))

        # Optional but recommended: speed station queries
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_velib_raw_station
            ON velib_raw (identifiant_du_site_de_comptage);
        """))

        # Optional: prevent duplicates (choose a key that matches your data reality)
        conn.execute(text("""
            CREATE UNIQUE INDEX IF NOT EXISTS uq_velib_station_time
            ON velib_raw (identifiant_du_site_de_comptage, date_et_heure_de_comptage);
        """))
    print("Schema ensured.", flush=True)



# def insert_velib_on_conflict_do_nothing(df: pd.DataFrame) -> int:
#     if df.empty:
#         return 0

#     cols = [
#         "identifiant_du_site_de_comptage",
#         "nom_du_site_de_comptage",
#         "comptage_horaire",
#         "date_et_heure_de_comptage",
#         "coordonnees_geographiques",
#     ]
#     df = df[cols].copy()

#     # ---- 1) Coerce numeric columns safely (vectorized) ----
#     # This yields float with NaN, then we keep as nullable Int64
#     df["identifiant_du_site_de_comptage"] = pd.to_numeric(
#         df["identifiant_du_site_de_comptage"], errors="coerce"
#     ).astype("Int64")

#     df["comptage_horaire"] = pd.to_numeric(
#         df["comptage_horaire"], errors="coerce"
#     ).astype("Int64")

#     # ---- 2) Coerce datetime safely ----
#     df["date_et_heure_de_comptage"] = pd.to_datetime(
#         df["date_et_heure_de_comptage"], format="%Y-%m-%dT%H:%M:%S%z", errors="coerce"
#     ) # 2025-10-19T20:00:00+00:00

#     # ---- 3) Replace pandas missing values with None (IMPORTANT) ----
#     # Do this AFTER coercions to catch NaN/NaT created by conversions
#     df = df.where(pd.notna(df), None)

#     # ---- 4) Ensure psycopg2-friendly Python objects ----
#     # Nullable Int64 can still carry pandas scalars; convert to object dtype
#     df["identifiant_du_site_de_comptage"] = df["identifiant_du_site_de_comptage"].astype(object)
#     df["comptage_horaire"] = df["comptage_horaire"].astype(object)
#     df["date_et_heure_de_comptage"] = df["date_et_heure_de_comptage"].astype(object)

#     # Optional: coordinates as string/None
#     df["coordonnees_geographiques"] = df["coordonnees_geographiques"].astype(object)
#     df["nom_du_site_de_comptage"] = df["nom_du_site_de_comptage"].astype(object)

#     records = list(df.itertuples(index=False, name=None))

#     sql = """
#         INSERT INTO velib_raw (
#             identifiant_du_site_de_comptage,
#             nom_du_site_de_comptage,
#             comptage_horaire,
#             date_et_heure_de_comptage,
#             coordonnees_geographiques
#         )
#         VALUES %s
#         ON CONFLICT (identifiant_du_site_de_comptage, date_et_heure_de_comptage)
#         DO NOTHING;
#     """

#     conn = psycopg2.connect(os.getenv("DATABASE_URL"))
#     try:
#         with conn, conn.cursor() as cur:
#             execute_values(cur, sql, records, page_size=2000)
#         return len(records)
#     finally:
#         conn.close()


def download_and_insert_in_chunks(engine, chunksize=50_000):
    url = csv_url()
    print(f"Downloading CSV from {url} ...", flush=True)

    # 1) Ensure schema exists with correct types
    ensure_velib_raw_schema(engine)

    csv_path = _download_to_tempfile(url, retries=4, timeout=(5, 180))

    try:
        print(f"Reading chunks from {csv_path} ...", flush=True)

        chunk_iter = pd.read_csv(
            csv_path,
            sep=";",
            chunksize=chunksize,
            usecols=[
                "Identifiant du site de comptage",
                "Nom du site de comptage",
                "Comptage horaire",
                "Date et heure de comptage",
                "Coordonnées géographiques",
            ],
        )

        total_rows = 0

        for i, chunk in enumerate(chunk_iter, start=1):
            rows = len(chunk)
            
            print(f"Inserting chunk {i} with at most {rows} rows in it assuming no duplicates.", flush=True)

            # 2) Standardize columns (your existing function)
            chunk = standardize_columns(chunk)

            # 3) Rename the coordinates column to match DB schema
            # After standardize_columns, it will likely be "coordonnées_géographiques" (still accented)
            # Normalize it once here:
            if "coordonnées_géographiques" in chunk.columns:
                print("Renaming 'coordonnées_géographiques' column to 'coordonnees_geographiques' ...", flush=True)
                chunk = chunk.rename(columns={"coordonnées_géographiques": "coordonnees_geographiques"})

            # 4) Coerce types so they match the table schema
            chunk = coerce_velib_types(chunk)
            # Drop rows with missing key columns (cannot be used for training/features)
            before = len(chunk)
            chunk = chunk.dropna(subset=["identifiant_du_site_de_comptage", "date_et_heure_de_comptage"])
            dropped = before - len(chunk)
            if dropped:
                print(f"Dropped {dropped} rows with missing station_id or datetime in chunk {i}", flush=True)
                
            # 5) Append only (do not replace)
            # insert_velib_on_conflict_do_nothing(chunk)
            inserted = insert_on_conflict_do_nothing(engine, chunk)
            print(f"Inserted {inserted} new rows for chunk {i}.", flush=True)
            total_rows += inserted
            print(f"Total rows inserted so far: {total_rows}.", flush=True)
            # chunk.to_sql(
            #     "velib_raw",
            #     engine,
            #     if_exists="append",
            #     index=False,
            #     method="multi",
            #     chunksize=2000,
            # )

        print(f"All chunks inserted successfully. Total rows inserted: {total_rows}", flush=True)

    finally:
        try:
            os.remove(csv_path)
        except OSError:
            pass


# ==============================
# Utilities
# ==============================

def wait_for_db(max_retries=10, delay=3):
    """
    Wait until PostgreSQL is ready.
    """
    for i in range(max_retries):
        try:
            engine = create_engine(database_url())
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            print("Database is ready.")
            return engine
        except OperationalError:
            print("Waiting for database...")
            time.sleep(delay)

    raise Exception("Database not available after retries.")


def table_exists_and_not_empty(table_name, engine):
    """
    Check if table exists and contains data.
    """
    with engine.connect() as conn:
        # Check if table exists
        result = conn.execute(text(f"""
            SELECT EXISTS (
                SELECT FROM information_schema.tables
                WHERE table_name = :table_name
            );
        """), {"table_name": table_name})

        exists = result.scalar()

        if not exists:
            return False

        # Check if table has rows
        result = conn.execute(text(f"SELECT COUNT(*) FROM {table_name};"))
        count = result.scalar()

        return count > 0
    
# ==============================
# Main
# ==============================

def main():
    if database_url() is None:
        print("DATABASE_URL not set.")
        sys.exit(1)

    if csv_url() is None:
        print("CSV_URL not set.")
        sys.exit(1)

    engine = wait_for_db()

    if table_exists_and_not_empty(table_name(), engine):
        print("Table already populated. Nothing to do.")
        return

    download_and_insert_in_chunks(engine)