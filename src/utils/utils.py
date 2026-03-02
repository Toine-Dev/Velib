from sqlalchemy import create_engine, inspect, text
from utils.config import database_url
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy import Table, MetaData

# -------------------------------
# Utility: Get max date from DB
# -------------------------------
def get_max_date(table_name: str, column_name: str):
    engine = create_engine(database_url())
    with engine.connect() as conn:
        inspector = inspect(engine)
        if not inspector.has_table(table_name):
            return None, None
        result_min = conn.execute(
            text(f"SELECT MIN({column_name}) FROM {table_name}")
        )        
        result_max = conn.execute(
            text(f"SELECT MAX({column_name}) FROM {table_name}")
        )
        return result_min.scalar(), result_max.scalar()
    







def drop_unique_index_if_exists(engine):
    with engine.begin() as conn:
        conn.execute(text("DROP INDEX IF EXISTS uq_velib_station_time;"))

def dedupe_velib_raw(engine):
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS velib_raw_dedup;"))

        conn.execute(text("""
            CREATE TABLE velib_raw_dedup AS
            SELECT DISTINCT ON (identifiant_du_site_de_comptage, date_et_heure_de_comptage)
                *
            FROM velib_raw
            WHERE identifiant_du_site_de_comptage IS NOT NULL
              AND date_et_heure_de_comptage IS NOT NULL
            ORDER BY
              identifiant_du_site_de_comptage,
              date_et_heure_de_comptage;
        """))

        conn.execute(text("DROP TABLE velib_raw;"))
        conn.execute(text("ALTER TABLE velib_raw_dedup RENAME TO velib_raw;"))

def create_unique_index(engine):
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE UNIQUE INDEX IF NOT EXISTS uq_velib_station_time
            ON velib_raw (identifiant_du_site_de_comptage, date_et_heure_de_comptage);
        """))

def create_supporting_indexes(engine):
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_velib_station
            ON velib_raw (identifiant_du_site_de_comptage);
        """))
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_velib_date
            ON velib_raw (date_et_heure_de_comptage);
        """))

def delete_duplicates(engine):
    drop_unique_index_if_exists(engine)
    dedupe_velib_raw(engine)
    create_unique_index(engine)
    create_supporting_indexes(engine)






def insert_on_conflict_do_nothing(engine, df, page_size=2000):
    if df.empty:
        return 0

    metadata = MetaData()
    velib_table = Table("velib_raw", metadata, autoload_with=engine)

    records = df.to_dict(orient="records")

    total_inserted = 0

    # Insert in smaller batches (important for large chunks)
    for i in range(0, len(records), page_size):
        batch = records[i:i+page_size]

        stmt = insert(velib_table).values(batch)

        stmt = stmt.on_conflict_do_nothing(
            index_elements=[
                "identifiant_du_site_de_comptage",
                "date_et_heure_de_comptage",
            ]
        )

        with engine.begin() as conn:
            result = conn.execute(stmt)
            total_inserted += result.rowcount

    return total_inserted