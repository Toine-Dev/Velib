import streamlit as st
import pandas as pd
import folium
import matplotlib.pyplot as plt
from folium.plugins import HeatMap
from streamlit_folium import folium_static
from datetime import timedelta
import requests
import os

API_BASE_URL = os.environ.get("API_BASE_URL", "http://api:8000")
API_USER = os.environ.get("STREAMLIT_API_USER", os.environ.get("API_ADMIN_USER", "admin"))
API_PASS = os.environ.get("STREAMLIT_API_PASS", os.environ.get("API_ADMIN_PASS", "admin"))

@st.cache_resource
def _get_api_token() -> str:
    """
    Fetch and cache a JWT token from the API.
    Uses OAuth2PasswordRequestForm => needs x-www-form-urlencoded.
    """
    r = requests.post(
        f"{API_BASE_URL}/auth/token",
        data={"username": API_USER, "password": API_PASS},
        timeout=15,
    )
    r.raise_for_status()
    return r.json()["access_token"]

def _api_headers() -> dict:
    return {"Authorization": f"Bearer {_get_api_token()}"}

def _api_get(path: str, params: dict | None = None):
    r = requests.get(f"{API_BASE_URL}{path}", headers=_api_headers(), params=params, timeout=30)
    r.raise_for_status()
    return r.json()

def _api_post(path: str, json_body: dict):
    r = requests.post(f"{API_BASE_URL}{path}", headers=_api_headers(), json=json_body, timeout=30)
    r.raise_for_status()
    return r.json()

# -------------------------------------------------------CACHING -------------------------------------------------------------------
@st.cache_data
def forecast_window():
    data = _api_get("/client/forecast-window")
    start_dt = pd.to_datetime(data["start"])
    end_dt = pd.to_datetime(data["end"])
    return start_dt, end_dt

@st.cache_data
def load_forecast_range(start_dt: pd.Timestamp, end_dt: pd.Timestamp) -> pd.DataFrame:
    data = _api_get(
        "/client/forecast-range",
        params={"start": start_dt.isoformat(), "end": end_dt.isoformat()},
    )
    df = pd.DataFrame(data)
    if df.empty:
        return df
    df["date_et_heure_de_comptage"] = pd.to_datetime(df["date_et_heure_de_comptage"])
    return df

@st.cache_data
def load_mlflow_info():
    """
    Charge depuis MLflow :
      - les métriques du meilleur run (MAE, RMSE, R²)
      - les feature importances (artifact feature_importances.csv ou .json)
    Retourne (metrics_dict | None, fi_df | DataFrame vide).
    """
    try:
        import mlflow

        tracking_uri   = os.environ.get("MLFLOW_TRACKING_URI",   "http://mlflow:5000")
        experiment_name = os.environ.get("MLFLOW_EXPERIMENT_NAME", "velib_forecast")
        mlflow.set_tracking_uri(tracking_uri)

        client     = mlflow.tracking.MlflowClient()
        experiment = client.get_experiment_by_name(experiment_name)
        if experiment is None:
            return None, pd.DataFrame()

        # Meilleur run trié par RMSE croissant
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["metrics.RMSE ASC"],
            max_results=1,
        )
        if not runs:
            return None, pd.DataFrame()

        best_run = runs[0]
        m = best_run.data.metrics

        metrics = {k: m.get(k) for k in ("MAE", "RMSE", "R2")}
        metrics = {k: v for k, v in metrics.items() if v is not None}
        if not metrics:
            metrics = None

        # Feature importances : artifact feature_importances.csv ou .json
        fi_df    = pd.DataFrame()
        run_id   = best_run.info.run_id
        art_names = [a.path for a in client.list_artifacts(run_id)]

        if "feature_importances.csv" in art_names:
            local = mlflow.artifacts.download_artifacts(
                run_id=run_id, artifact_path="feature_importances.csv"
            )
            fi_df = pd.read_csv(local)

        elif "feature_importances.json" in art_names:
            import json
            local = mlflow.artifacts.download_artifacts(
                run_id=run_id, artifact_path="feature_importances.json"
            )
            with open(local) as f:
                fi_df = pd.DataFrame(json.load(f))

        # Normalisation des noms de colonnes → "feature" / "importance"
        if not fi_df.empty:
            fi_df.columns = [c.lower() for c in fi_df.columns]
            col_map = {}
            for c in fi_df.columns:
                if any(k in c for k in ("feature", "name")):
                    col_map[c] = "feature"
                elif any(k in c for k in ("import", "value", "score")):
                    col_map[c] = "importance"
            fi_df = fi_df.rename(columns=col_map)

            if {"feature", "importance"}.issubset(fi_df.columns):
                fi_df = (
                    fi_df[["feature", "importance"]]
                    .sort_values("importance", ascending=False)
                    .head(10)
                    .reset_index(drop=True)
                )
            else:
                fi_df = pd.DataFrame()

        return metrics, fi_df

    except Exception as e:
        st.warning(f"MLflow indisponible ou erreur lors du chargement : {e}")
        return None, pd.DataFrame()

@st.cache_data
def load_historical_reference(site_id: int, target_dt: pd.Timestamp) -> pd.DataFrame:
    data = _api_get(
        "/client/historical-reference",
        params={"site_id": site_id, "datetime": target_dt.isoformat()},
    )
    return pd.DataFrame(data)

def _threshold_color(value: float, t_low: int, t_high: int) -> str:
    if value < t_low:
        return "#00cc44"   # vert
    elif value < t_high:
        return "#ffcc00"   # jaune
    else:
        return "#cc0000"   # rouge
    
def build_heatmap(df: pd.DataFrame, radius: int, t_low: int, t_high: int) -> folium.Map:
    m = folium.Map(location=[48.8566, 2.3522], zoom_start=12, tiles="CartoDB positron")

    df_clean = df[
        ["latitude", "longitude", "comptage_horaire", "nom_du_site_de_comptage", "identifiant_du_site_de_comptage"]
    ].dropna().copy()

    cap = max(t_high * 2, float(df_clean["comptage_horaire"].max()))
    df_clean["weight"] = df_clean["comptage_horaire"].clip(upper=cap) / cap

    low_norm  = max(0.01, min(round(t_low  / cap, 3), 0.49))
    high_norm = max(low_norm + 0.01, min(round(t_high / cap, 3), 0.99))

    HeatMap(
        data=df_clean[["latitude", "longitude", "weight"]].values.tolist(),
        radius=radius,
        max_zoom=13,
        min_opacity=0.3,
        gradient={
            "0.0":                   "#00cc44",
            str(low_norm):           "#00cc44",
            str(low_norm  + 0.001):  "#ffcc00",
            str(high_norm):          "#ffcc00",
            str(high_norm + 0.001):  "#cc0000",
            "1.0":                   "#cc0000",
        },
    ).add_to(m)

    for _, row in df_clean.iterrows():
        color = _threshold_color(row["comptage_horaire"], t_low, t_high)
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=6,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.75,
            weight=1,
            tooltip=folium.Tooltip(
                f"<b>{row['nom_du_site_de_comptage']}</b>"
                f"<br>ID : {int(row['identifiant_du_site_de_comptage'])}"
                f"<br>Prévision : <b>{row['comptage_horaire']:.0f} vélos</b>",
                sticky=False,
            ),
        ).add_to(m)

    legend_html = f"""
    <div style="
        position:fixed; bottom:30px; left:30px; z-index:1000;
        background:white; padding:10px 14px; border-radius:8px;
        box-shadow:0 2px 6px rgba(0,0,0,.3); font-size:13px; line-height:1.9;">
      <b>Vélos prévus</b><br>
      <span style="color:#00cc44;font-size:18px">●</span> &lt; {t_low} &nbsp;(calme)<br>
      <span style="color:#ffcc00;font-size:18px">●</span> {t_low} – {t_high} &nbsp;(modéré)<br>
      <span style="color:#cc0000;font-size:18px">●</span> &gt; {t_high} &nbsp;(chargé)
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    return m



# def build_heatmap(df: pd.DataFrame, radius: int) -> folium.Map:
#     """
#     Carte Folium avec :
#     - HeatMap à gradient dynamique (intensité = min/max de l'heure sélectionnée)
#     - CircleMarkers transparents avec tooltip nom + comptage
#     - Légende
#     """
#     m = folium.Map(location=[48.8566, 2.3522], zoom_start=12, tiles="CartoDB positron")

#     df_clean = df[
#         ["latitude", "longitude", "comptage_horaire", "nom_du_site_de_comptage", "identifiant_du_site_de_comptage"]
#     ].dropna()

#     df_clean["site_label"] = df_clean.apply(
#         lambda r: f"{r['nom_du_site_de_comptage']} (ID {int(r['identifiant_du_site_de_comptage'])})",
#         axis=1,
#     )

#     vmin = float(df_clean["comptage_horaire"].min())
#     vmax = float(df_clean["comptage_horaire"].max())
#     if vmax == vmin:
#         vmax = vmin + 1

#     # Poids normalisés 0→1 pour la HeatMap
#     df_clean["weight"] = (df_clean["comptage_horaire"] - vmin) / (vmax - vmin)

#     HeatMap(
#         data=df_clean[["latitude", "longitude", "weight"]].values.tolist(),
#         radius=radius,
#         max_zoom=13,
#         min_opacity=0.3,
#         gradient={
#             "0.0": "#00cc44",   # vert  = peu de vélos
#             "0.5": "#ffcc00",   # jaune = moyen
#             "1.0": "#cc0000",   # rouge = beaucoup
#         },
#     ).add_to(m)

#     # Markers invisibles portant les tooltips
#     for _, row in df_clean.iterrows():
#         color = _value_to_hex(row["comptage_horaire"], vmin, vmax)
#         folium.CircleMarker(
#             location=[row["latitude"], row["longitude"]],
#             radius=6,
#             color=color,
#             fill=True,
#             fill_color=color,
#             fill_opacity=0.7,
#             weight=1,
#             tooltip=folium.Tooltip(
#                 f"<b>{row['nom_du_site_de_comptage']}</b><br>"
#                 f"Prévision : <b>{row['comptage_horaire']:.0f} vélos</b>",
#                 sticky=False,
#             ),
#         ).add_to(m)

#     # Légende
#     legend_html = f"""
#     <div style="
#         position:fixed; bottom:30px; left:30px; z-index:1000;
#         background:white; padding:10px 14px; border-radius:8px;
#         box-shadow:0 2px 6px rgba(0,0,0,.3); font-size:13px; line-height:1.8;">
#       <b>Vélos prévus</b><br>
#       <span style="color:#00cc44;font-size:18px">●</span> Peu &nbsp;({vmin:.0f})<br>
#       <span style="color:#ffcc00;font-size:18px">●</span> Moyen<br>
#       <span style="color:#cc0000;font-size:18px">●</span> Beaucoup &nbsp;({vmax:.0f})
#     </div>
#     """
#     m.get_root().html.add_child(folium.Element(legend_html))

#     return m

# -------------------------------------------------------PAGE MODEL PREDICTIONS-------------------------------------------------------------------

def show_prediction():
    st.title("🚲 Prédictions de fréquentation Vélib'")

    start_dt, end_dt = forecast_window()

    # ── Métriques & Feature importances (depuis MLflow) ─────────────────────────
    with st.expander("📊 Performances du modèle & Features importantes", expanded=True):
        metrics, fi_df = load_mlflow_info()

        if metrics:
            st.subheader("Métriques de validation")
            col1, col2, col3 = st.columns(3)
            if "MAE"  in metrics: col1.metric("MAE",  f"{metrics['MAE']:.4f}")
            if "RMSE" in metrics: col2.metric("RMSE", f"{metrics['RMSE']:.4f}")
            if "R2"   in metrics: col3.metric("R²",   f"{metrics['R2']:.4f}")
        else:
            st.info("Aucune métrique MLflow disponible.")

        if not fi_df.empty:
            st.subheader("Top 10 features importantes")
            fig, ax = plt.subplots(figsize=(8, 5))
            fi_sorted = fi_df.sort_values("importance", ascending=True)
            ax.barh(fi_sorted["feature"], fi_sorted["importance"], color="steelblue")
            ax.set_xlabel("Importance")
            ax.set_title("Top 10 features importantes")
            st.pyplot(fig)
        else:
            st.info("Aucune feature importance disponible dans MLflow.")

    st.divider()

    # ── Sélecteur date / heure ──────────────────────────────────────────────────
    st.subheader("🗓️ Choisissez une date et une heure de prévision")
    st.info(
        f"Fenêtre disponible : **{start_dt.strftime('%d/%m/%Y %H:%M')}** "
        f"→ **{end_dt.strftime('%d/%m/%Y %H:%M')}**"
    )

    col1, col2 = st.columns(2)
    pred_date = col1.date_input(
        "Date",
        min_value=start_dt.date(),
        max_value=end_dt.date(),
        value=start_dt.date(),
    )
    pred_time = col2.time_input("Heure", value=start_dt.time())

    pred_time = pred_time.replace(minute=0, second=0, microsecond=0)
    dt = pd.Timestamp.combine(pred_date, pred_time)

    if dt < start_dt or dt > end_dt:
        st.error(
            f"⚠️ Veuillez choisir une date/heure dans la fenêtre : "
            f"{start_dt.strftime('%d/%m/%Y %H:%M')} → {end_dt.strftime('%d/%m/%Y %H:%M')}"
        )
        st.stop()

    radius = st.slider("Rayon de la heatmap", min_value=10, max_value=50, value=30)

    st.markdown("**Paliers de couleur (vélos prévus)**")
    col_s1, col_s2 = st.columns(2)
    t_low  = col_s1.slider("🟡 Seuil vert → jaune",  min_value=10, max_value=500, value=50,  step=10)
    t_high = col_s2.slider("🔴 Seuil jaune → rouge", min_value=10, max_value=500, value=150, step=10)
    if t_high <= t_low:
        st.warning("⚠️ Le seuil rouge doit être supérieur au seuil jaune.")
        t_high = t_low + 10

    # ── Bouton principal ────────────────────────────────────────────────────────
    if st.button("🔍 Prédire la heatmap", type="primary"):
        # df_hour = load_forecast_geo(dt)
        rows = _api_post("/client/predict", {"datetime": dt.strftime("%Y-%m-%d %H:%M:%S")})
        df_hour = pd.DataFrame(rows)
        if df_hour.empty:
            st.error("Aucune prévision trouvée pour cette heure. (velib_forecast_geo vide ?)")
            st.stop()

        df_range = load_forecast_range(start_dt, end_dt)

        st.session_state["df_hour"]  = df_hour
        st.session_state["df_range"] = df_range
        st.session_state["pred_dt"]  = dt

    # ── Résultats (persistants après clic) ──────────────────────────────────────
    if "df_hour" not in st.session_state:
        st.stop()

    df_hour  = st.session_state["df_hour"]
    dt       = st.session_state["pred_dt"]
    df_range = st.session_state["df_range"]

    # ── Heatmap ─────────────────────────────────────────────────────────────────
    st.subheader(f"🗺️ Heatmap des prévisions — {dt.strftime('%d/%m/%Y %H:%M')}")
    st.caption(
        f"🟢 &lt; {t_low} vélos (calme) &nbsp;→&nbsp; "
        f"🟡 {t_low}–{t_high} (modéré) &nbsp;→&nbsp; "
        f"🔴 &gt; {t_high} (chargé) &nbsp;|&nbsp; "
        "Survolez un point pour voir le nom de la station."
    )
    folium_static(build_heatmap(df_hour, radius, t_low, t_high))

    st.divider()

    # ── Tableaux Top 15 ─────────────────────────────────────────────────────────
    col_avoid, col_go = st.columns(2)

    with col_avoid:
        st.subheader("🔴 Top 15 — À éviter")
        st.caption("Stations avec le plus de vélos prévus")
        top_busy = (
            df_hour.assign(
                Station=lambda d: d.apply(
                    lambda r: f"{r['nom_du_site_de_comptage']} (ID {int(r['identifiant_du_site_de_comptage'])})",
                    axis=1,
                )
            )[["Station", "comptage_horaire"]]
            .sort_values("comptage_horaire", ascending=False)
            .head(15)
            .reset_index(drop=True)
        )
        top_busy.index += 1
        st.dataframe(
            top_busy.rename(columns={"comptage_horaire": "Vélos prévus"}),
            use_container_width=True,
        )

    with col_go:
        st.subheader("🟢 Top 15 — Tranquilles")
        st.caption("Stations avec le moins de vélos prévus")
        top_quiet = (
            df_hour.assign(
                Station=lambda d: d.apply(
                    lambda r: f"{r['nom_du_site_de_comptage']} (ID {int(r['identifiant_du_site_de_comptage'])})",
                    axis=1,
                )
            )[["Station", "comptage_horaire"]]
            .sort_values("comptage_horaire", ascending=True)
            .head(15)
            .reset_index(drop=True)
        )
        top_quiet.index += 1
        st.dataframe(
            top_quiet.rename(columns={"comptage_horaire": "Vélos prévus"}),
            use_container_width=True,
        )

    st.divider()

    # ── Graphique d'évolution par station sur 48h ────────────────────────────────
    st.subheader("📈 Évolution horaire d'une station sur 48h")

    site_choice = None
    site_id_choice = None
    site_data = pd.DataFrame()

    if df_range.empty:
        st.warning("Impossible de charger les données d'évolution sur 48h.")
    else:
        # Sécurise les données : une seule ligne par station + heure
        df_range = (
            df_range.sort_values(
                ["identifiant_du_site_de_comptage", "date_et_heure_de_comptage"]
            )
            .drop_duplicates(
                subset=["identifiant_du_site_de_comptage", "date_et_heure_de_comptage"],
                keep="first",
            )
            .copy()
        )

        # Libellé lisible pour l'UI
        df_range["site_label"] = df_range.apply(
            lambda r: f"{r['nom_du_site_de_comptage']} (ID {int(r['identifiant_du_site_de_comptage'])})",
            axis=1,
        )

        # Mapping unique station_id -> label
        site_options_df = (
            df_range[
                ["identifiant_du_site_de_comptage", "nom_du_site_de_comptage", "site_label"]
            ]
            .drop_duplicates(subset=["identifiant_du_site_de_comptage"])
            .sort_values(["nom_du_site_de_comptage", "identifiant_du_site_de_comptage"])
            .reset_index(drop=True)
        )

        site_labels = site_options_df["site_label"].tolist()

        # Pré-sélection = station la plus chargée à l'heure choisie
        default_site_id = (
            int(
                df_hour.sort_values("comptage_horaire", ascending=False)
                .iloc[0]["identifiant_du_site_de_comptage"]
            )
            if not df_hour.empty
            else int(site_options_df.iloc[0]["identifiant_du_site_de_comptage"])
        )

        default_idx_matches = site_options_df.index[
            site_options_df["identifiant_du_site_de_comptage"] == default_site_id
        ].tolist()
        default_idx = default_idx_matches[0] if default_idx_matches else 0

        selected_label = st.selectbox(
            "Choisissez un site de comptage :",
            options=site_labels,
            index=default_idx,
        )

        selected_row = site_options_df.loc[
            site_options_df["site_label"] == selected_label
        ].iloc[0]

        site_id_choice = int(selected_row["identifiant_du_site_de_comptage"])
        site_choice = selected_row["nom_du_site_de_comptage"]

        site_data = (
            df_range[df_range["identifiant_du_site_de_comptage"] == site_id_choice]
            .sort_values("date_et_heure_de_comptage")
            .set_index("date_et_heure_de_comptage")
        )

        st.line_chart(
            site_data[["comptage_horaire"]].rename(
                columns={"comptage_horaire": "Vélos prévus"}
            )
        )

        # ── Comparaison à l'heure sélectionnée ──────────────────────────────────────
    st.subheader(f"📌 Comparaison à {dt.strftime('%A %d/%m %H:%M')}")
    st.caption("Valeurs historiques issues de `velib_raw` — même jour de semaine, même heure.")

    pred_val = None
    if not site_data.empty and dt in site_data.index:
        selected_val = site_data.loc[dt, "comptage_horaire"]
        if isinstance(selected_val, pd.Series):
            pred_val = float(selected_val.iloc[0])
        else:
            pred_val = float(selected_val)

    hist_df = (
        load_historical_reference(site_id_choice, dt)
        if site_id_choice is not None
        else pd.DataFrame()
    )

    n_total = 1 + len(hist_df)
    metric_cols = st.columns(n_total)

    with metric_cols[0]:
        st.metric(
            label="🔮 Prévision modèle",
            value=f"{pred_val:.0f} vélos" if pred_val is not None else "N/A",
        )

    for i, row in hist_df.iterrows():
        with metric_cols[i + 1]:
            delta = (
                f"{pred_val - row['comptage_horaire']:+.0f} vs prévu"
                if pred_val is not None
                else None
            )
            st.metric(
                label=f"📅 {row['label']}",
                value=f"{row['comptage_horaire']:.0f} vélos",
                delta=delta,
                delta_color="inverse",
            )

    if hist_df.empty:
        st.info("Aucune donnée historique disponible pour cette station / cette période.")




