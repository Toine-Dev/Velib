import streamlit as st
import pandas as pd
import folium
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from folium.plugins import HeatMap
from streamlit_folium import folium_static
from datetime import timedelta
from data.loader import load_forecast_geo, get_engine
import os

# -------------------------------------------------------CACHING -------------------------------------------------------------------

@st.cache_data
def forecast_window():
    engine = get_engine()
    last_raw = pd.read_sql(
        "SELECT MAX(date_et_heure_de_comptage) AS dt FROM velib_raw;", engine
    ).loc[0, "dt"]
    last_raw = pd.to_datetime(last_raw).floor("H")
    return last_raw + timedelta(hours=1), last_raw + timedelta(hours=48)

@st.cache_data
def load_forecast_range(start_dt: pd.Timestamp, end_dt: pd.Timestamp) -> pd.DataFrame:
    """Charge toutes les prévisions entre start_dt et end_dt depuis velib_forecast_geo."""
    engine = get_engine()
    query = """
        SELECT
            date_et_heure_de_comptage,
            nom_du_site_de_comptage,
            latitude,
            longitude,
            comptage_horaire
        FROM velib_forecast_geo
        WHERE date_et_heure_de_comptage BETWEEN %(start)s AND %(end)s
        ORDER BY nom_du_site_de_comptage, date_et_heure_de_comptage;
    """
    df = pd.read_sql(query, engine, params={"start": start_dt, "end": end_dt})
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
def load_historical_reference(site: str, target_dt: pd.Timestamp) -> pd.DataFrame:
    """
    Pour un site et une heure cible, charge depuis velib_raw :
      - Moyenne des 4 dernières semaines (même jour de semaine, même heure)
    """
    engine = get_engine()
    target_hour = target_dt.hour
    target_dow  = target_dt.dayofweek  # 0=lundi…6=dimanche

    rows = []

    # Moyenne des 4 semaines (même jour de semaine, même heure)
    avg_query = """
        SELECT AVG(comptage_horaire) AS val
        FROM velib_raw
        WHERE nom_du_site_de_comptage = %(site)s
          AND EXTRACT(ISODOW FROM date_et_heure_de_comptage) = %(dow)s
          AND EXTRACT(HOUR   FROM date_et_heure_de_comptage) = %(hour)s
          AND date_et_heure_de_comptage BETWEEN %(ws)s AND %(we)s;
    """
    avg_res = pd.read_sql(avg_query, engine, params={
        "site": site,
        "dow":  target_dow + 1,   # ISODOW : 1=lundi…7=dimanche
        "hour": target_hour,
        "ws":   target_dt - timedelta(weeks=4),
        "we":   target_dt - timedelta(hours=1),
    })
    avg_val = avg_res.loc[0, "val"] if not avg_res.empty and avg_res.loc[0, "val"] is not None else None
    if avg_val is not None:
        rows.append({
            "label": "Moy. 4 semaines (même jour/heure)",
            "comptage_horaire": round(float(avg_val), 1),
        })

    return pd.DataFrame(rows)

def _value_to_hex(value: float, vmin: float, vmax: float) -> str:
    """Valeur → couleur hex sur gradient vert→jaune→rouge."""
    cmap = plt.get_cmap("RdYlGn_r")
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    r, g, b, _ = cmap(norm(np.clip(value, vmin, vmax)))
    return "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))

def build_heatmap(df: pd.DataFrame, radius: int) -> folium.Map:
    """
    Carte Folium avec :
    - HeatMap à gradient dynamique (intensité = min/max de l'heure sélectionnée)
    - CircleMarkers transparents avec tooltip nom + comptage
    - Légende
    """
    m = folium.Map(location=[48.8566, 2.3522], zoom_start=12, tiles="CartoDB positron")

    df_clean = df[["latitude", "longitude", "comptage_horaire", "nom_du_site_de_comptage"]].dropna().copy()

    vmin = float(df_clean["comptage_horaire"].min())
    vmax = float(df_clean["comptage_horaire"].max())
    if vmax == vmin:
        vmax = vmin + 1

    # Poids normalisés 0→1 pour la HeatMap
    df_clean["weight"] = (df_clean["comptage_horaire"] - vmin) / (vmax - vmin)

    HeatMap(
        data=df_clean[["latitude", "longitude", "weight"]].values.tolist(),
        radius=radius,
        max_zoom=13,
        min_opacity=0.3,
        gradient={
            "0.0": "#00cc44",   # vert  = peu de vélos
            "0.5": "#ffcc00",   # jaune = moyen
            "1.0": "#cc0000",   # rouge = beaucoup
        },
    ).add_to(m)

    # Markers invisibles portant les tooltips
    for _, row in df_clean.iterrows():
        color = _value_to_hex(row["comptage_horaire"], vmin, vmax)
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=6,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            weight=1,
            tooltip=folium.Tooltip(
                f"<b>{row['nom_du_site_de_comptage']}</b><br>"
                f"Prévision : <b>{row['comptage_horaire']:.0f} vélos</b>",
                sticky=False,
            ),
        ).add_to(m)

    # Légende
    legend_html = f"""
    <div style="
        position:fixed; bottom:30px; left:30px; z-index:1000;
        background:white; padding:10px 14px; border-radius:8px;
        box-shadow:0 2px 6px rgba(0,0,0,.3); font-size:13px; line-height:1.8;">
      <b>Vélos prévus</b><br>
      <span style="color:#00cc44;font-size:18px">●</span> Peu &nbsp;({vmin:.0f})<br>
      <span style="color:#ffcc00;font-size:18px">●</span> Moyen<br>
      <span style="color:#cc0000;font-size:18px">●</span> Beaucoup &nbsp;({vmax:.0f})
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    return m

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

    # ── Bouton principal ────────────────────────────────────────────────────────
    if st.button("🔍 Prédire la heatmap", type="primary"):
        df_hour = load_forecast_geo(dt)
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
        "🟢 Peu de vélos → 🟡 Moyen → 🔴 Beaucoup &nbsp;|&nbsp; "
        "Couleurs relatives au min/max de l'heure sélectionnée. "
        "Survolez un point pour voir le nom de la station."
    )
    folium_static(build_heatmap(df_hour, radius))

    st.divider()

    # ── Tableaux Top 15 ─────────────────────────────────────────────────────────
    col_avoid, col_go = st.columns(2)

    with col_avoid:
        st.subheader("🔴 Top 15 — À éviter")
        st.caption("Stations avec le plus de vélos prévus")
        top_busy = (
            df_hour[["nom_du_site_de_comptage", "comptage_horaire"]]
            .sort_values("comptage_horaire", ascending=False)
            .head(15)
            .reset_index(drop=True)
        )
        top_busy.index += 1
        st.dataframe(
            top_busy.rename(columns={
                "nom_du_site_de_comptage": "Station",
                "comptage_horaire": "Vélos prévus",
            }),
            use_container_width=True,
        )

    with col_go:
        st.subheader("🟢 Top 15 — Tranquilles")
        st.caption("Stations avec le moins de vélos prévus")
        top_quiet = (
            df_hour[["nom_du_site_de_comptage", "comptage_horaire"]]
            .sort_values("comptage_horaire", ascending=True)
            .head(15)
            .reset_index(drop=True)
        )
        top_quiet.index += 1
        st.dataframe(
            top_quiet.rename(columns={
                "nom_du_site_de_comptage": "Station",
                "comptage_horaire": "Vélos prévus",
            }),
            use_container_width=True,
        )

    st.divider()

    # ── Graphique d'évolution par station sur 48h ────────────────────────────────
    st.subheader("📈 Évolution horaire d'une station sur 48h")

    if df_range.empty:
        st.warning("Impossible de charger les données d'évolution sur 48h.")
    else:
        sites = sorted(df_range["nom_du_site_de_comptage"].unique())

        # Pré-sélectionner la station la plus chargée à l'heure choisie
        default_site = (
            df_hour.sort_values("comptage_horaire", ascending=False)
            .iloc[0]["nom_du_site_de_comptage"]
            if not df_hour.empty
            else sites[0]
        )
        default_idx = sites.index(default_site) if default_site in sites else 0

        site_choice = st.selectbox(
            "Choisissez un site de comptage :",
            options=sites,
            index=default_idx,
        )

        site_data = (
            df_range[df_range["nom_du_site_de_comptage"] == site_choice]
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

    pred_val = float(site_data.loc[dt, "comptage_horaire"]) if dt in site_data.index else None
    hist_df  = load_historical_reference(site_choice, dt)

    n_total = 1 + len(hist_df)
    metric_cols = st.columns(n_total)

    with metric_cols[0]:
        st.metric(
            label="🔮 Prévision modèle",
            value=f"{pred_val:.0f} vélos" if pred_val is not None else "N/A",
        )

    for i, row in hist_df.iterrows():
        with metric_cols[i + 1]:
            delta = f"{pred_val - row['comptage_horaire']:+.0f} vs prévu" if pred_val is not None else None
            st.metric(
                label=f"📅 {row['label']}",
                value=f"{row['comptage_horaire']:.0f} vélos",
                delta=delta,
                delta_color="inverse",  # rouge si modèle prédit plus que l'historique
            )

    if hist_df.empty:
        st.info("Aucune donnée historique disponible pour cette station / cette période.")




# def show_prediction():

#     

#     pipeline = None
#     metrics = None

#     # Vérifier si le modèle existe
#     if os.path.exists("model.pkl"):
#         pipeline = load_model()
#         # model = pipeline.named_steps['model']
#         if os.path.exists("metrics.json"):
#             with open("metrics.json", "r") as f:
#                 metrics = json.load(f)
#     else:
#         st.info("Aucun modèle trouvé. Entraînement en cours...")
#         with st.spinner("Training model..."):
#             train_model()
#             pipeline = load_model()

#     # Afficher les métriques
#     if metrics:
#         st.write("Validation Metrics:")
#         col1, col2, col3 = st.columns(3)
#         col1.metric("MAE", f"{metrics['MAE']:.4f}")
#         col2.metric("RMSE", f"{metrics['RMSE']:.4f}")
#         col3.metric("R²", f"{metrics['R2']:.4f}")

# # # Feature Importance Plot
#     st.subheader("Top Features Importantes")
#     transformed_feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()
#     feature_importances = pipeline.named_steps['model'].feature_importances_
#     importances_df = pd.DataFrame({'Feature': transformed_feature_names , 'Importance': feature_importances}).sort_values(by='Importance', ascending=False).head(10)
#     fig, ax = plt.subplots(figsize=(8,6))
#     ax.barh(importances_df['Feature'], importances_df['Importance'], color='skyblue')
#     ax.set_xlabel('Importance')
#     ax.set_ylabel('Feature')
#     ax.set_title('Top 10 features importantes')
#     ax.invert_yaxis()
#     st.pyplot(fig)

# # --- Section prédiction interactive ---
#     start_datetime, start_date = last_cached_datetime()
#     st.header("Prédiction pour une date future")
#     st.write(f"Veuillez choisir une date/heure STRICTEMENT après {start_datetime}.")
#     col1, col2 = st.columns(2)

#     pred_date = col1.date_input("Sélectionnez la date", min_value=start_date)
#     pred_time = col2.time_input("Sélectionnez l'heure")
#     # The tz_localize method does not convert the time, it just assigns the timezone.
#     # pred_datetime = pd.Timestamp.combine(pred_date, pred_time).tz_localize(start_datetime.tz)
#     pred_datetime = pd.Timestamp.combine(pred_date, pred_time)

#     if pred_datetime <= start_datetime or pred_datetime > (start_datetime + timedelta(hours=48)):
#         st.error(f"Veuillez choisir une combinaison date et heure STRICTEMENT après {start_datetime} mais pas plus de 48 heures après cette dernière.")
#         st.stop()

#     radius = st.slider("Rayon de la heatmap", min_value=10, max_value=50, value=30)

#     if st.button("Prédire la heatmap"):
#         pred_time = pred_time.replace(minute=0, second=0, microsecond=0)
#         datetime_pred = pd.Timestamp.combine(pred_date, pred_time)
#         locations = predict_model(datetime_pred)
        
#         # Heatmap
#         m = folium.Map(location=[48.8566, 2.3522], zoom_start=12, tiles="CartoDB positron")
#         HeatMap(data=locations[['latitude', 'longitude', 'comptage_horaire']].values.tolist(), radius=radius, max_zoom=13).add_to(m)
#         st.subheader("Heatmap des prédictions")
#         folium_static(m)

#         # Stocker pour graphique
#         st.session_state['locations'] = locations

#     # Graphique après heatmap
#     if 'locations' in st.session_state:
#         st.subheader("Choisissez un site pour voir son évolution")
#         site_choice = st.selectbox("Site :", st.session_state['locations']['nom_du_site_de_comptage'].unique())
#         site_data = st.session_state['locations'][st.session_state['locations']['nom_du_site_de_comptage'] == site_choice]
#         site_data_sorted = site_data.sort_values('date_et_heure_de_comptage')
#         st.line_chart(site_data_sorted[['date_et_heure_de_comptage', 'comptage_horaire']].set_index('date_et_heure_de_comptage'))