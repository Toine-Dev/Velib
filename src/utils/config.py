import os

# ==============================
#### API endpoints ####
# ==============================

def get_required(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"{name} is not set")
    return value

def forecast_weather_api_url() -> str:
    return get_required("FORECAST_WEATHER_API_URL")

def historical_weather_api_url() -> str:
    return get_required("HISTORICAL_WEATHER_API_URL")

def velib_api_url() -> str:
    return get_required("VELIB_API_URL")

# ==============================
#### Database
# ==============================

def database_url() -> str:
    return get_required("DATABASE_URL")

def table_name() -> str:
    return get_required("TABLE_NAME")

def csv_url() -> str:
    return get_required("CSV_URL")