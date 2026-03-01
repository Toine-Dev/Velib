from utils.config import csv_url, database_url, table_name
from data.ingestion import download_and_insert_in_chunks
from sqlalchemy import create_engine, text
import sys
import time
from sqlalchemy.exc import OperationalError

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

    print("Data pipeline completed successfully.")