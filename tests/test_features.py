"""
Tests unitaires — models/features.py
Couvre : get_season_from_date, is_rush_hour, is_vacances, is_night,
         add_cyclic_features, add_weather_features, compute_site_stats,
         build_future_timeframe, build_history
"""
import pytest
import pandas as pd
import numpy as np
import sys, pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))

from models.features import (
    get_season_from_date,
    is_rush_hour,
    is_vacances,
    is_night,
    add_cyclic_features,
    add_weather_features,
    compute_site_stats,
    build_future_timeframe,
    build_history,
)


# ===========================================================================
# 1. Shared date helpers (features.py reimplements them — test independently)
# ===========================================================================
class TestSeasonHelper:
    @pytest.mark.parametrize("date,expected", [
        ("2025-04-01", "spring"),
        ("2025-07-15", "summer"),
        ("2025-10-10", "autumn"),
        ("2025-01-05", "winter"),
        ("2025-12-22", "winter"),
    ])
    def test_season_mapping(self, date, expected):
        assert get_season_from_date(date) == expected

    def test_tz_aware_input(self):
        ts = pd.Timestamp("2025-08-01 10:00", tz="UTC")
        assert get_season_from_date(ts) == "summer"


class TestRushHourHelper:
    @pytest.mark.parametrize("hour,expected", [
        (7, True), (9, True), (10, False),
        (17, True), (19, True), (20, False),
        (0, False), (12, False),
    ])
    def test_rush_hour_boundaries(self, hour, expected):
        ts = pd.Timestamp(f"2025-03-01 {hour:02d}:00:00")
        assert is_rush_hour(ts) == expected


class TestVacancesHelper:
    def test_in_holiday_period(self):
        assert is_vacances(pd.Timestamp("2025-02-20")) is True

    def test_outside_all_periods(self):
        assert is_vacances(pd.Timestamp("2025-03-15")) is False

    def test_exact_start_included(self):
        assert is_vacances(pd.Timestamp("2025-04-12")) is True

    def test_exact_end_excluded(self):
        assert is_vacances(pd.Timestamp("2025-04-29")) is False


class TestNightHelper:
    def _row(self, dt_str, season):
        return {"date_et_heure_de_comptage": pd.Timestamp(dt_str), "saison": season}

    def test_summer_late_night(self):
        assert is_night(self._row("2025-07-01 23:30:00", "summer")) is True

    def test_summer_afternoon(self):
        assert is_night(self._row("2025-07-01 15:00:00", "summer")) is False

    def test_winter_early_evening(self):
        assert is_night(self._row("2025-01-15 18:00:00", "winter")) is True


# ===========================================================================
# 2. add_weather_features
# ===========================================================================
class TestAddWeatherFeatures:
    def _df(self):
        return pd.DataFrame({
            "rain": [0.0, 3.5, None],
            "snowfall": [0.0, 0.0, 1.2],
            "wind_speed_10m": [5.0, 20.0, 10.0],
            "apparent_temperature": [15.0, 12.0, None],
        })

    def test_pluie_created(self):
        df = add_weather_features(self._df())
        assert "pluie" in df.columns

    def test_neige_created(self):
        df = add_weather_features(self._df())
        assert "neige" in df.columns

    def test_vent_created(self):
        df = add_weather_features(self._df())
        assert "vent" in df.columns

    def test_pluie_false_when_zero_rain(self):
        df = add_weather_features(self._df())
        assert bool(df["pluie"].iloc[0]) is False

    def test_pluie_true_when_positive_rain(self):
        df = add_weather_features(self._df())
        assert bool(df["pluie"].iloc[1]) is True

    def test_vent_true_above_15(self):
        df = add_weather_features(self._df())
        assert bool(df["vent"].iloc[1]) is True

    def test_nan_rain_treated_as_zero(self):
        df = add_weather_features(self._df())
        assert bool(df["pluie"].iloc[2]) is False

    def test_apparent_temperature_nan_filled(self):
        df = add_weather_features(self._df())
        assert not df["apparent_temperature"].isna().any()


# ===========================================================================
# 3. add_cyclic_features
# ===========================================================================
class TestAddCyclicFeaturesModels:
    def _df(self):
        return pd.DataFrame({
            "heure": [0, 6, 12, 18],
            "jour": [0, 1, 2, 6],
            "mois": [1, 3, 6, 12],
            "saison": ["winter", "spring", "summer", "autumn"],
        })

    def test_all_cyclic_columns_present(self):
        df = add_cyclic_features(self._df())
        expected = ["heure_sin", "heure_cos", "jour_sin", "jour_cos",
                    "mois_sin", "mois_cos", "saison_sin", "saison_cos"]
        for col in expected:
            assert col in df.columns

    def test_values_bounded(self):
        df = add_cyclic_features(self._df())
        for col in ["heure_sin", "heure_cos", "jour_sin", "jour_cos",
                    "mois_sin", "mois_cos", "saison_sin", "saison_cos"]:
            assert df[col].between(-1.0, 1.0).all()

    def test_no_nulls(self):
        df = add_cyclic_features(self._df())
        cyclic = ["heure_sin", "heure_cos", "jour_sin", "jour_cos",
                  "mois_sin", "mois_cos", "saison_sin", "saison_cos"]
        assert not df[cyclic].isna().any().any()

    def test_hour_12_cos_is_minus_one(self):
        # cos(2π * 12/24) = cos(π) = -1
        df = add_cyclic_features(self._df())
        row = df[self._df()["heure"] == 12]
        assert abs(row["heure_cos"].iloc[0] - (-1.0)) < 1e-9

    def test_hour_0_cos_is_one(self):
        df = add_cyclic_features(self._df())
        row = df[self._df()["heure"] == 0]
        assert abs(row["heure_cos"].iloc[0] - 1.0) < 1e-9


# ===========================================================================
# 4. compute_site_stats
# ===========================================================================
class TestComputeSiteStats:
    def _df(self):
        return pd.DataFrame({
            "identifiant_du_site_de_comptage": [1, 1, 1, 2, 2],
            "comptage_horaire": [10.0, 20.0, 30.0, 5.0, 15.0],
        })

    def test_output_columns(self):
        result = compute_site_stats(self._df())
        for col in ["site_mean_usage", "site_usage_variability",
                    "site_max_usage", "site_min_usage"]:
            assert col in result.columns

    def test_mean_correct(self):
        result = compute_site_stats(self._df())
        site1 = result[result["identifiant_du_site_de_comptage"] == 1]
        assert abs(site1["site_mean_usage"].iloc[0] - 20.0) < 1e-9

    def test_max_correct(self):
        result = compute_site_stats(self._df())
        site1 = result[result["identifiant_du_site_de_comptage"] == 1]
        assert site1["site_max_usage"].iloc[0] == 30.0

    def test_min_correct(self):
        result = compute_site_stats(self._df())
        site2 = result[result["identifiant_du_site_de_comptage"] == 2]
        assert site2["site_min_usage"].iloc[0] == 5.0

    def test_one_row_per_site(self):
        result = compute_site_stats(self._df())
        assert len(result) == 2


# ===========================================================================
# 5. build_future_timeframe
# ===========================================================================
class TestBuildFutureTimeframe:
    def test_returns_dataframe(self):
        start = pd.Timestamp("2025-03-01 10:00:00")
        end = pd.Timestamp("2025-03-01 13:00:00")
        df = build_future_timeframe(start, end)
        assert isinstance(df, pd.DataFrame)

    def test_correct_number_of_hours(self):
        start = pd.Timestamp("2025-03-01 10:00:00")
        end = pd.Timestamp("2025-03-01 13:00:00")
        df = build_future_timeframe(start, end)
        # inclusive="right" → excludes start, includes end → 3 rows
        assert len(df) == 3

    def test_start_excluded(self):
        start = pd.Timestamp("2025-03-01 10:00:00")
        end = pd.Timestamp("2025-03-01 12:00:00")
        df = build_future_timeframe(start, end)
        assert start not in df["date_et_heure_de_comptage"].values

    def test_end_included(self):
        start = pd.Timestamp("2025-03-01 10:00:00")
        end = pd.Timestamp("2025-03-01 12:00:00")
        df = build_future_timeframe(start, end)
        assert end in df["date_et_heure_de_comptage"].values

    def test_tz_stripped_from_output(self):
        start = pd.Timestamp("2025-03-01 10:00:00", tz="UTC")
        end = pd.Timestamp("2025-03-01 13:00:00", tz="UTC")
        df = build_future_timeframe(start, end)
        assert df["date_et_heure_de_comptage"].dt.tz is None

    def test_all_minutes_are_zero(self):
        start = pd.Timestamp("2025-03-01 10:00:00")
        end = pd.Timestamp("2025-03-01 14:00:00")
        df = build_future_timeframe(start, end)
        assert (df["date_et_heure_de_comptage"].dt.minute == 0).all()


# ===========================================================================
# 6. build_history
# ===========================================================================
class TestBuildHistory:
    def _processed_df(self):
        site_ids = [1] * 30 + [2] * 10
        times = pd.date_range("2025-01-01", periods=30, freq="h").tolist() + \
                pd.date_range("2025-01-01", periods=10, freq="h").tolist()
        counts = list(range(30)) + list(range(10, 20))
        return pd.DataFrame({
            "identifiant_du_site_de_comptage": site_ids,
            "date_et_heure_de_comptage": times,
            "comptage_horaire": counts,
        })

    def _unique_sites(self, processed_df):
        return processed_df[["identifiant_du_site_de_comptage"]].drop_duplicates()

    def test_returns_dict(self):
        df = self._processed_df()
        result = build_history(df, self._unique_sites(df))
        assert isinstance(result, dict)

    def test_all_sites_present(self):
        df = self._processed_df()
        result = build_history(df, self._unique_sites(df))
        assert 1 in result and 2 in result

    def test_history_length_is_24(self):
        df = self._processed_df()
        result = build_history(df, self._unique_sites(df))
        for site, hist in result.items():
            assert len(hist) == 24, f"Site {site}: expected 24, got {len(hist)}"

    def test_short_site_padded_to_24(self):
        # Site 2 has only 10 rows → should be padded
        df = self._processed_df()
        result = build_history(df, self._unique_sites(df))
        assert len(result[2]) == 24

    def test_last_values_are_most_recent(self):
        df = self._processed_df()
        result = build_history(df, self._unique_sites(df))
        # Site 1 has 30 values (0..29); last 24 are 6..29
        assert result[1][-1] == 29