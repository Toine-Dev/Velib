from sqlalchemy import create_engine
from sqlalchemy import text
from data.ingestion import update_velib, update_weather, update_weather_forecast, upsert_velib_sites
from utils.config import database_url
from utils.utils import delete_duplicates

### use this to delete last 7 days from TODAY of data in velib_raw to test update pipeline ####
engine = create_engine(database_url())

TABLE_NAME = "velib_raw"

# Compute the cutoff (7 days ago from now)
with engine.begin() as conn:
    result = conn.execute(
        text("""
            DELETE FROM velib_raw
            WHERE date_et_heure_de_comptage::timestamptz
                  >= NOW() - INTERVAL '7 days'
        """)
    )
    print(f"Deleted {result.rowcount} rows")



## This will have to be implemented to ensure uniqueness of records ###
# def ensure_constraints(engine):
#     with engine.begin() as conn:  # auto-commit/rollback
#         conn.execute(text("""
#             CREATE UNIQUE INDEX IF NOT EXISTS velib_unique
#             ON velib_raw (identifiant_du_site_de_comptage, date_et_heure_de_comptage);
#         """))

#         conn.execute(text("""
#             CREATE UNIQUE INDEX IF NOT EXISTS weather_unique
#             ON weather_raw ("time");
#         """))

# def ensure_constraints():
#     with engine.connect() as conn:
#         conn.execute(text("""
#             ALTER TABLE velib_raw
#             ADD CONSTRAINT IF NOT EXISTS velib_unique
#             UNIQUE (identifiant_du_site_de_comptage, date_et_heure_de_comptage);
#         """))

#         conn.execute(text("""
#             ALTER TABLE weather_raw
#             ADD CONSTRAINT IF NOT EXISTS weather_unique
#             UNIQUE (time);
#         """))

#         conn.commit()



if __name__ == "__main__":
    engine = create_engine(database_url())
    update_velib(engine)
    update_weather(engine)
    update_weather_forecast(engine)
    upsert_velib_sites(engine)
    # delete_duplicates(engine)
    print("✅ Pipeline complete.")