from sqlalchemy import create_engine, inspect, text
from utils.config import database_url

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