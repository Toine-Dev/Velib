"""
Tests unitaires — data/preprocessing.py
Couvre : standardize_columns, coerce_velib_types, get_season_from_date,
         is_night, is_vacances, is_rush_hour, add_cyclic_features,
         preprocess_velib_data, preprocess_weather_data
"""
import pytest
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Helpers pour reconstruire les fonctions sans dépendances DB
# ---------------------------------------------------------------------------
import sys, pathlib
# Permet d'importer sans avoir toute l'arbo installée
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))

from data.preprocessing import (
    standardize_columns,
    coerce_velib_types,
    get_season_from_date,
    is_night,
    is_vacances,
    is_rush_hour,
    add_cyclic_features,
    preprocess_velib_data,
    preprocess_weather_data,
)


# ===========================================================================
# 1. standardize_columns
# ===========================================================================
class TestStandardizeColumns:
    def test_spaces_replaced_by_underscore(self):
        df = pd.DataFrame(columns=["Col A", "Col B"])
        result = standardize_columns(df)
        assert list(result.columns) == ["col_a", "col_b"]

    def test_already_clean_columns_unchanged(self):
        df = pd.DataFrame(columns=["col_a", "col_b"])
        result = standardize_columns(df)
        assert list(result.columns) == ["col_a", "col_b"]

    def test_uppercase_lowercased(self):
        df = pd.DataFrame(columns=["ID", "NAME"])
        result = standardize_columns(df)
        assert list(result.columns) == ["id", "name"]

    def test_leading_trailing_spaces_stripped(self):
        df = pd.DataFrame(columns=["  col a  "])
        result = standardize_columns(df)
        assert list(result.columns) == ["col_a"]


# ===========================================================================
# 2. coerce_velib_types
# ===========================================================================
class TestCoerceVelibTypes:
    def _base_df(self):
        return pd.DataFrame({
            "identifiant_du_site_de_comptage": ["100056789", "100056790"],
            "comptage_horaire": ["42", "17"],
            "date_et_heure_de_comptage": ["2025-03-01 08:00:00", "2025-03-01 09:00:00"],
        })

    def test_site_id_becomes_int64(self):
        df = coerce_velib_types(self._base_df())
        assert pd.api.types.is_integer_dtype(df["identifiant_du_site_de_comptage"])

    def test_comptage_horaire_becomes_int64(self):
        df = coerce_velib_types(self._base_df())
        assert pd.api.types.is_integer_dtype(df["comptage_horaire"])

    def test_datetime_parsed_correctly(self):
        df = coerce_velib_types(self._base_df())
        assert pd.api.types.is_datetime64_any_dtype(df["date_et_heure_de_comptage"])

    def test_invalid_site_id_becomes_na(self):
        df = self._base_df()
        df.loc[0, "identifiant_du_site_de_comptage"] = "not_a_number"
        result = coerce_velib_types(df)
        assert pd.isna(result.loc[0, "identifiant_du_site_de_comptage"])

    def test_invalid_datetime_becomes_nat(self):
        df = self._base_df()
        df.loc[0, "date_et_heure_de_comptage"] = "not_a_date"
        result = coerce_velib_types(df)
        assert pd.isna(result.loc[0, "date_et_heure_de_comptage"])


# ===========================================================================
# 3. get_season_from_date
# ===========================================================================
class TestGetSeasonFromDate:
    def test_spring(self):
        assert get_season_from_date("2025-04-15") == "spring"

    def test_summer(self):
        assert get_season_from_date("2025-07-20") == "summer"

    def test_autumn(self):
        assert get_season_from_date("2025-10-01") == "autumn"

    def test_winter_january(self):
        assert get_season_from_date("2025-01-15") == "winter"

    def test_winter_december(self):
        assert get_season_from_date("2025-12-25") == "winter"

    def test_spring_boundary(self):
        # 20 March = first day of spring
        assert get_season_from_date("2025-03-20") == "spring"

    def test_tz_aware_input_handled(self):
        ts = pd.Timestamp("2025-07-01 12:00:00", tz="Europe/Paris")
        assert get_season_from_date(ts) == "summer"

    def test_returns_string(self):
        result = get_season_from_date("2025-06-01")
        assert isinstance(result, str)

    def test_valid_season_values(self):
        dates = ["2025-01-01", "2025-04-01", "2025-07-01", "2025-10-01"]
        valid = {"spring", "summer", "autumn", "winter"}
        for d in dates:
            assert get_season_from_date(d) in valid


# ===========================================================================
# 4. is_night
# ===========================================================================
class TestIsNight:
    def _row(self, dt_str: str, season: str):
        return {
            "date_et_heure_de_comptage": pd.Timestamp(dt_str),
            "saison": season,
        }

    def test_winter_night_late_evening(self):
        row = self._row("2025-01-15 20:00:00", "winter")
        assert is_night(row) is True

    def test_winter_day(self):
        row = self._row("2025-01-15 12:00:00", "winter")
        assert is_night(row) is False

    def test_summer_night_midnight(self):
        row = self._row("2025-07-15 23:00:00", "summer")
        assert is_night(row) is True

    def test_summer_daytime(self):
        row = self._row("2025-07-15 14:00:00", "summer")
        assert is_night(row) is False

    def test_unknown_season_returns_false(self):
        row = self._row("2025-07-15 23:00:00", "monsoon")
        assert is_night(row) is False


# ===========================================================================
# 5. is_vacances
# ===========================================================================
class TestIsVacances:
    def test_toussaint_in_vacances(self):
        assert is_vacances(pd.Timestamp("2024-10-25")) is True

    def test_noel_in_vacances(self):
        assert is_vacances(pd.Timestamp("2024-12-25")) is True

    def test_hiver_in_vacances(self):
        assert is_vacances(pd.Timestamp("2025-02-20")) is True

    def test_regular_day_not_vacances(self):
        assert is_vacances(pd.Timestamp("2025-03-10")) is False

    def test_summer_start_in_vacances(self):
        assert is_vacances(pd.Timestamp("2025-07-10")) is True

    def test_end_boundary_excluded(self):
        # End date is exclusive (< end), so 2025-09-02 should NOT be vacances
        assert is_vacances(pd.Timestamp("2025-09-02")) is False

    def test_start_boundary_included(self):
        assert is_vacances(pd.Timestamp("2024-10-19")) is True


# ===========================================================================
# 6. is_rush_hour
# ===========================================================================
class TestIsRushHour:
    def test_morning_rush(self):
        assert is_rush_hour(pd.Timestamp("2025-03-01 08:00:00")) is True

    def test_evening_rush(self):
        assert is_rush_hour(pd.Timestamp("2025-03-01 18:00:00")) is True

    def test_midday_not_rush(self):
        assert is_rush_hour(pd.Timestamp("2025-03-01 13:00:00")) is False

    def test_night_not_rush(self):
        assert is_rush_hour(pd.Timestamp("2025-03-01 02:00:00")) is False

    def test_boundary_7am_is_rush(self):
        assert is_rush_hour(pd.Timestamp("2025-03-01 07:00:00")) is True

    def test_boundary_10am_not_rush(self):
        assert is_rush_hour(pd.Timestamp("2025-03-01 10:00:00")) is False

    def test_boundary_17h_is_rush(self):
        assert is_rush_hour(pd.Timestamp("2025-03-01 17:00:00")) is True

    def test_boundary_20h_not_rush(self):
        assert is_rush_hour(pd.Timestamp("2025-03-01 20:00:00")) is False


# ===========================================================================
# 7. add_cyclic_features
# ===========================================================================
class TestAddCyclicFeatures:
    def _base_df(self):
        return pd.DataFrame({
            "jour": [0, 3, 6],       # Mon, Thu, Sun
            "mois": [1, 6, 12],
            "heure": [0, 12, 23],
            "saison": ["winter", "summer", "autumn"],
            "date_et_heure_de_comptage": pd.to_datetime([
                "2025-01-06 00:00:00",
                "2025-06-15 12:00:00",
                "2025-10-05 23:00:00",
            ]),
        })

    def test_cyclic_columns_created(self):
        df = add_cyclic_features(self._base_df())
        for col in ["jour_sin", "jour_cos", "mois_sin", "mois_cos",
                    "heure_sin", "heure_cos", "saison_sin", "saison_cos"]:
            assert col in df.columns, f"Missing column: {col}"

    def test_original_time_columns_dropped(self):
        df = add_cyclic_features(self._base_df())
        for col in ["jour", "mois", "heure", "saison"]:
            assert col not in df.columns

    def test_cyclic_values_in_minus1_plus1(self):
        df = add_cyclic_features(self._base_df())
        for col in ["heure_sin", "heure_cos", "jour_sin", "jour_cos"]:
            assert df[col].between(-1.0, 1.0).all(), f"{col} out of [-1, 1]"

    def test_no_nan_in_cyclic_features(self):
        df = add_cyclic_features(self._base_df())
        cyclic_cols = ["jour_sin", "jour_cos", "mois_sin", "mois_cos",
                       "heure_sin", "heure_cos", "saison_sin", "saison_cos"]
        assert not df[cyclic_cols].isna().any().any()

    def test_hour_0_sin_is_zero(self):
        df = pd.DataFrame({
            "jour": [0], "mois": [1], "heure": [0],
            "saison": ["winter"],
            "date_et_heure_de_comptage": pd.to_datetime(["2025-01-06 00:00:00"]),
        })
        result = add_cyclic_features(df)
        assert abs(result["heure_sin"].iloc[0]) < 1e-10


# ===========================================================================
# 8. preprocess_velib_data
# ===========================================================================
class TestPreprocessVelibData:
    def _minimal_df(self):
        return pd.DataFrame({
            "identifiant_du_site_de_comptage": [100056789, 100056789, 100056789],
            "comptage_horaire": [10, 20, 15],
            "date_et_heure_de_comptage": [
                "2025-03-01 08:00:00",
                "2025-03-01 09:00:00",
                "2025-03-01 10:00:00",
            ],
        })

    def test_output_has_expected_feature_columns(self):
        df = preprocess_velib_data(self._minimal_df())
        for col in ["heure", "mois", "jour", "saison", "vacances",
                    "heure_de_pointe", "nuit"]:
            assert col in df.columns, f"Missing column: {col}"

    def test_rows_with_null_site_dropped(self):
        df = self._minimal_df()
        df.loc[0, "identifiant_du_site_de_comptage"] = None
        result = preprocess_velib_data(df)
        assert len(result) == 2

    def test_rows_with_null_comptage_dropped(self):
        df = self._minimal_df()
        df.loc[1, "comptage_horaire"] = None
        result = preprocess_velib_data(df)
        assert len(result) == 2

    def test_datetime_floored_to_hour(self):
        df = self._minimal_df()
        df.loc[0, "date_et_heure_de_comptage"] = "2025-03-01 08:45:00"
        result = preprocess_velib_data(df)
        assert result["date_et_heure_de_comptage"].iloc[0].minute == 0

    def test_tz_stripped_from_datetime(self):
        df = self._minimal_df()
        df["date_et_heure_de_comptage"] = pd.to_datetime(
            df["date_et_heure_de_comptage"]
        ).dt.tz_localize("Europe/Paris")
        result = preprocess_velib_data(df)
        assert result["date_et_heure_de_comptage"].dt.tz is None

    def test_vacances_column_is_boolean(self):
        df = preprocess_velib_data(self._minimal_df())
        assert df["vacances"].dtype == bool or df["vacances"].dtype == object

    def test_heure_de_pointe_correct_at_8am(self):
        df = preprocess_velib_data(self._minimal_df())
        # 08:00 should be rush hour
        row_8am = df[df["date_et_heure_de_comptage"].dt.hour == 8]
        assert row_8am["heure_de_pointe"].iloc[0] is True or \
               row_8am["heure_de_pointe"].iloc[0] == True


# ===========================================================================
# 9. preprocess_weather_data
# ===========================================================================
class TestPreprocessWeatherData:
    def _minimal_weather_df(self):
        return pd.DataFrame({
            "time": ["2025-03-01 08:00:00", "2025-03-01 09:00:00"],
            "rain": [0.0, 5.2],
            "wind_speed_10m": [10.0, 20.0],
            "snowfall": [0.0, 0.0],
            "apparent_temperature": [12.0, 11.5],
        })

    def test_pluie_column_created(self):
        df = preprocess_weather_data(self._minimal_weather_df())
        assert "pluie" in df.columns

    def test_vent_column_created(self):
        df = preprocess_weather_data(self._minimal_weather_df())
        assert "vent" in df.columns

    def test_neige_column_created(self):
        df = preprocess_weather_data(self._minimal_weather_df())
        assert "neige" in df.columns

    def test_pluie_true_when_rain_positive(self):
        df = preprocess_weather_data(self._minimal_weather_df())
        assert df["pluie"].iloc[1] is True or df["pluie"].iloc[1] == True

    def test_pluie_false_when_no_rain(self):
        df = preprocess_weather_data(self._minimal_weather_df())
        assert df["pluie"].iloc[0] is False or df["pluie"].iloc[0] == False

    def test_vent_true_when_speed_above_15(self):
        df = preprocess_weather_data(self._minimal_weather_df())
        assert df["vent"].iloc[1] is True or df["vent"].iloc[1] == True

    def test_time_floored_to_hour(self):
        wdf = self._minimal_weather_df()
        wdf.loc[0, "time"] = "2025-03-01 08:37:00"
        df = preprocess_weather_data(wdf)
        assert df["time"].iloc[0].minute == 0