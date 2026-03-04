import streamlit as st
import os
from ui.pages.overview import show_overview
from ui.pages.analysis import show_analysis
from ui.pages.prediction import show_prediction

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
github_logo = os.path.join(BASE_DIR, "images", "logo_github.jpg")
paris_logo = os.path.join(BASE_DIR, "images", "logo_ville_paris.png")

#------------------------------------------------------- SIDE BAR-------------------------------------------------------------------
with st.sidebar:

    # ---------------- Navigation ----------------
    page = st.radio(
        "Navigation",
        [
            "Overview",
            "Data Analysis",
            "Model & Predictions"
        ],
        index=0
    )

    # ---------------- Bloc Auteurs ----------------
    st.markdown("""
    <div style="
        background-color: #f0f4fa;
        padding: 10px 15px;
        border-radius: 8px;
        font-size: 0.9em;">
        <b>Auteurs :</b><br>
        Antoine Scarcella<br>
        Nathan Vitse<br>
        Nikhil Teja Bellamkonda
    </div>
    """, unsafe_allow_html=True)

    st.markdown("")

    # ---------------- Données ----------------
    st.markdown("**Données :**")

    col1, col2 = st.columns([1, 4])

    with col1:
        st.image(paris_logo, width=50)

    with col2:
        st.markdown(
            "[Open Data Ville de Paris](https://opendata.paris.fr/explore/dataset/comptage-velo-donnees-compteurs/information/)",
        )

    # ---------------- Séparateur ----------------
    st.markdown("---")

    # ---------------- GitHub ----------------
    col3, col4 = st.columns([1, 4])

    with col3:
        st.image(github_logo, width=50)

    with col4:
        st.markdown(
            "[Voir notre GitHub](https://github.com/Toine-Dev/Velib)"
        )

#-------------------------------------------------------PAGE CONFIG-------------------------------------------------------------------
st.set_page_config(
    page_title="Dashboard afflluence vélos à Paris",
    layout="wide"
)

st.markdown("""
    <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            padding-left: 3rem;
            padding-right: 3rem;
            max-width: 1400px;
            margin: auto;
        }
    </style>
""", unsafe_allow_html=True)

if page == "Overview":
    show_overview()

elif page == "Data Analysis":
    show_analysis()

elif page == "Model & Predictions":
    show_prediction()