from __future__ import annotations
from data.preprocessing import preprocess_merged_data, preprocess_velib_data, preprocess_weather_data
from utils.config import database_url
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

# --- Your existing functions ---
# def preprocess_velib_data(raw_velib_df: pd.DataFrame) -> pd.DataFrame: ...
# def preprocess_weather_data(raw_weather_df: pd.DataFrame) -> pd.DataFrame: ...

# def database_url() -> str:
#     # return "postgresql+psycopg://user:pass@host:5432/dbname"
#     raise NotImplementedError

VELIB_RAW_TABLE = "velib_raw"
WEATHER_RAW_TABLE = "weather_raw"
OUT_TABLE = "velib_weather_processed"

ID_COL = "identifiant_du_site_de_comptage"
VELIB_TIME_COL = "date_et_heure_de_comptage"
WEATHER_TIME_COL = "time"


def load_and_preprocess_weather(engine: Engine) -> pd.DataFrame:
    raw_weather_df = pd.read_sql_query(
        text(f"SELECT * FROM {WEATHER_RAW_TABLE}"),
        con=engine,
    )
    processed_weather_data = preprocess_weather_data(raw_weather_df)

    # Safety: ensure datetime and timezone consistency
    processed_weather_data[WEATHER_TIME_COL] = pd.to_datetime(processed_weather_data[WEATHER_TIME_COL], utc=True, errors="coerce")

    # Optional: drop duplicates on time to make merge deterministic (keep last)
    processed_weather_data = (
        processed_weather_data
        .dropna(subset=[WEATHER_TIME_COL])
        .sort_values(WEATHER_TIME_COL)
        .drop_duplicates(subset=[WEATHER_TIME_COL], keep="last")
        .reset_index(drop=True)
    )

    return processed_weather_data


def iter_site_ids(engine: Engine) -> list:
    # Pull distinct site ids from velib_raw
    df_sites = pd.read_sql_query(
        text(f"SELECT DISTINCT {ID_COL} AS site_id FROM {VELIB_RAW_TABLE} ORDER BY {ID_COL}"),
        con=engine,
    )
    return df_sites["site_id"].tolist()


def read_velib_for_site(engine: Engine, site_id, columns: list[str] | None = None) -> pd.DataFrame:
    # Select only needed columns for speed/memory
    col_sql = ", ".join(columns) if columns else "*"

    # Important: order by time if your preprocessing needs time series continuity
    q = text(f"""
        SELECT {col_sql}
        FROM {VELIB_RAW_TABLE}
        WHERE {ID_COL} = :site_id
        ORDER BY {VELIB_TIME_COL} ASC
    """)
    return pd.read_sql_query(q, con=engine, params={"site_id": site_id})


def write_processed(engine: Engine, df: pd.DataFrame, if_exists: str = "append") -> None:
    # Use multi inserts + chunking for speed
    df.to_sql(
        OUT_TABLE,
        con=engine,
        if_exists=if_exists,
        index=False,
        method="multi",
        chunksize=10_000, # Starting from Python 3.6+, you can use underscores in numeric literals to improve readability. They’re completely ignored by the interpreter.
        # If we had chunksize=100000 instead, it would compile to the same integer. No behavioral difference at all.
    )


def run_site_by_site_etl(
    engine: Engine,
    velib_columns: list[str] | None = None,
    truncate_output_first: bool = True,
) -> None:
    processed_weather_data = load_and_preprocess_weather(engine)
    site_ids = iter_site_ids(engine)

    # Optional: start fresh each run
    if truncate_output_first:
        with engine.begin() as conn:
            conn.execute(text(f"DROP TABLE IF EXISTS {OUT_TABLE}"))

    first_write = True

    for i, site_id in enumerate(site_ids, 1):
        raw_velib_df = read_velib_for_site(engine, site_id, columns=velib_columns)

        if raw_velib_df.empty:
            print(f"[{i}/{len(site_ids)}] site={site_id}: no rows, skipping")
            continue

        processed_velib_data = preprocess_velib_data(raw_velib_df)

        # Safety: ensure datetime and timezone consistency
        processed_velib_data[VELIB_TIME_COL] = pd.to_datetime(processed_velib_data[VELIB_TIME_COL], utc=True, errors="coerce")
        processed_velib_data = processed_velib_data.dropna(subset=[VELIB_TIME_COL])

        df_merged = (
            pd.merge(
                processed_velib_data,
                processed_weather_data,
                how="left",
                left_on=VELIB_TIME_COL,
                right_on=WEATHER_TIME_COL,
            )
            .drop(columns=[WEATHER_TIME_COL], errors="ignore")
        )

        df_encoded, _ = preprocess_merged_data(df_merged)

        # Write incrementally
        # write_processed(engine, df_merged, if_exists="replace" if first_write else "append")
        write_processed(engine, df_encoded, if_exists="replace" if first_write else "append")
        first_write = False

        print(f"[{i}/{len(site_ids)}] site={site_id}: raw={len(raw_velib_df):,} processed={len(df_merged):,} written")


def main() -> None:
    engine = create_engine(database_url(), pool_pre_ping=True)

    # # Strongly recommended: select only columns you actually need for preprocessing
    # velib_columns = [
    #     ID_COL,
    #     VELIB_TIME_COL,
    #     # add the rest you need...
    #     # "comptage_horaire",
    #     # "etat_station",
    #     # ...
    # ]

    # run_site_by_site_etl(engine, velib_columns=velib_columns, truncate_output_first=True)
    run_site_by_site_etl(engine, truncate_output_first=True) # Will drop the output table each time it is run to start fresh.


if __name__ == "__main__":
    main()


# processed_velib_data = preprocess_velib_data(raw_velib_df)
# processed_weather_data = preprocess_weather_data(raw_weather_df)
# df_merged = pd.merge(processed_velib_data, processed_weather_data, how="left", left_on="date_et_heure_de_comptage", right_on="time").drop(columns=["time"])
# processed_df, feature_names = preprocess_merged_data(df_merged)