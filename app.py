import streamlit as st
from pages.overview import show_overview
from pages.analysis import show_analysis
from pages.prediction import show_prediction

# ------------------------------ PATHS ------------------------------
# CSV_PATH = "comptage_velo_donnees_compteurs.csv"

#------------------------------------------------------- SIDE BAR-------------------------------------------------------------------
with st.sidebar:
    page = st.radio(
        "Navigation",
        [
            "Overview",
            "Data Analysis",
            "Model & Predictions"
        ],
        index=0
    )

    st.markdown("""
    <div style="
        background-color: #f0f4fa;
        padding: 10px 15px;
        border-radius: 8px;
        font-size: 0.9em;
        ">
        <b>Auteurs :</b><br>
Antoine Scarcella<br>
Nathan Vitse<br>
Nikhil Teja Bellamkonda<br><br>

<b>Données :
<a href="https://opendata.paris.fr/explore/dataset/comptage-velo-donnees-compteurs/download/?format=csv&timezone=Europe/Paris&lang=fr&use_labels_for_header=true&csv_separator=%3B" target="_self" style="color:#0066cc; text-decoration:none;">
data.gouv.fr
</a>
</div>
""", unsafe_allow_html=True)

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

# st.dataframe(raw_df.head())
# st.write(raw_df.columns)