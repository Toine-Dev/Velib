from db.db import main
from utils.config import database_url
from sqlalchemy import create_engine, text
from utils.utils import delete_duplicates, insert_on_conflict_do_nothing

engine = create_engine(database_url())

if __name__ == "__main__":
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS velib_raw;")) # Use this to drop the table and start fresh with the CSV data.
    main()
    print("Data pipeline completed successfully.")