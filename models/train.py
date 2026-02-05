import json
import os
import pandas as pd
from utils.config import load_model, make_preprocessor, train_final_model, save_model
from data.loader import load_processed_data, load_raw_velib_data, load_raw_weather_data
import streamlit as st
import matplotlib.pyplot as plt
from utils.config import DATA, MODELS

def train_model():
    pipeline = None
    metrics = None

    raw_df_velib = load_raw_velib_data()
    raw_df_weather = load_raw_weather_data()
    processed_df, feature_names = load_processed_data(raw_df_velib, raw_df_weather)

    # Vérifier si le modèle existe
    if os.path.exists("model.pkl"):
        pipeline = load_model()
        model = pipeline.named_steps['model']
        if os.path.exists("metrics.json"):
            with open("metrics.json", "r") as f:
                metrics = json.load(f)
    else:
        st.info("Aucun modèle trouvé. Entraînement en cours...")
        with st.spinner("Training model..."):
            target_cols = ['identifiant_du_site_de_comptage']
            numeric_cols = [col for col in feature_names if col not in target_cols]
            # Préprocesseur
            preprocessor = make_preprocessor(target_cols, numeric_cols)

            model_params = {
                "n_estimators" : 500, 
                "learning_rate" : 0.05,
                "max_depth" : -1,
                "random_state" : 42
            }

            pipeline, metrics = train_final_model(processed_df[feature_names], processed_df[['comptage_horaire']], model_params, target_cols, numeric_cols, test_size_ratio=0.1)
            save_model(pipeline)

            with open("metrics.json", "w") as f:
                json.dump(metrics, f)
            st.success("Modèle entraîné et sauvegardé dans model.pkl")

    # Afficher les métriques
    if metrics:
        st.write("Validation Metrics:")
        col1, col2, col3 = st.columns(3)
        col1.metric("MAE", f"{metrics['MAE']:.4f}")
        col2.metric("RMSE", f"{metrics['RMSE']:.4f}")
        col3.metric("R²", f"{metrics['R2']:.4f}")

    # Feature Importance Plot
    st.subheader("Top Features Importantes")
    transformed_feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()
    feature_importances = pipeline.named_steps['model'].feature_importances_
    importances_df = pd.DataFrame({'Feature': transformed_feature_names , 'Importance': feature_importances}).sort_values(by='Importance', ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(8,6))
    ax.barh(importances_df['Feature'], importances_df['Importance'], color='skyblue')
    ax.set_xlabel('Importance')
    ax.set_ylabel('Feature')
    ax.set_title('Top 10 features importantes')
    ax.invert_yaxis()
    st.pyplot(fig)

if __name__ == "__main__":
    train_model()