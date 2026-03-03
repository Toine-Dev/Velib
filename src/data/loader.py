import os
import pandas as pd
from sqlalchemy import create_engine, text
import streamlit as st

DB_URL = os.getenv("DATABASE_URL")

@st.cache_resource
def get_engine():
    return create_engine(DB_URL)


@st.cache_data
def load_raw_velib_data(limit: int | None = None):
    engine = get_engine()

    q = "SELECT * FROM velib_raw ORDER BY date_et_heure_de_comptage DESC"
    if limit:
        q += f" LIMIT {limit}"

    df = pd.read_sql(q, engine)
    df["date_et_heure_de_comptage"] = pd.to_datetime(df["date_et_heure_de_comptage"])
    return df


@st.cache_data
def load_raw_weather_data(limit: int | None = None):
    engine = get_engine()

    q = "SELECT * FROM weather_raw ORDER BY time DESC"
    if limit:
        q += f" LIMIT {limit}"

    df = pd.read_sql(q, engine)
    df["time"] = pd.to_datetime(df["time"])
    return df


@st.cache_data
def load_processed_data():
    engine = get_engine()

    df = pd.read_sql("SELECT * FROM velib_weather_processed", engine)
    df["date_et_heure_de_comptage"] = pd.to_datetime(df["date_et_heure_de_comptage"])

    return df


@st.cache_data
def load_forecast_geo(selected_datetime):
    engine = get_engine()

    query = text("""
        SELECT *
        FROM velib_forecast_geo
        WHERE date_et_heure_de_comptage = :dt
    """)

    df = pd.read_sql(query, engine, params={"dt": selected_datetime})
    return df