import streamlit as st
import pandas as pd
import folium
import matplotlib.pyplot as plt
from folium.plugins import HeatMap
from streamlit_folium import folium_static
from datetime import timedelta
from data.loader import load_forecast_geo, get_engine
import os

# -------------------------------------------------------CACHING -------------------------------------------------------------------

@st.cache_data
def forecast_window():
    """Retourne (start, end) de la fenêtre de prévision disponible."""
    engine = get_engine()
    last_raw = pd.read_sql(
        "SELECT MAX(date_et_heure_de_comptage) AS dt FROM velib_raw;", engine
    ).loc[0, "dt"]
    last_raw = pd.to_datetime(last_raw).floor("H")
    start = last_raw + timedelta(hours=1)
    end   = last_raw + timedelta(hours=48)
    return start, end


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
    m = folium.Map(location=[48.8566, 2.3522], zoom_start=12, tiles="CartoDB positron")
    heat_data = (
        df_hour[["latitude", "longitude", "comptage_horaire"]]
        .dropna()
        .values.tolist()
    )
    HeatMap(data=heat_data, radius=radius, max_zoom=13).add_to(m)
    folium_static(m)

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

        # Valeur ponctuelle à l'heure sélectionnée
        if dt in site_data.index:
            val = site_data.loc[dt, "comptage_horaire"]
            st.metric(
                label=f"Prévision à {dt.strftime('%H:%M')} le {dt.strftime('%d/%m/%Y')}",
                value=f"{val:.0f} vélos",
            )




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