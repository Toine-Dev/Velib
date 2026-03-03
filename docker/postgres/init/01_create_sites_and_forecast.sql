-- 01_create_sites_and_forecast.sql
-- Creates velib_sites dimension, velib_forecast fact table and a useful view.
-- Run once during DB initialization (mounted into /docker-entrypoint-initdb.d/).

-- 1) sites dimension table

CREATE TABLE IF NOT EXISTS velib_sites (
    identifiant_du_site_de_comptage BIGINT PRIMARY KEY,
    nom_du_site_de_comptage TEXT,
    latitude DOUBLE PRECISION,
    longitude DOUBLE PRECISION
);

-- 2) forecast table (stores latest forecasts; we'll delete a horizon on each run)
CREATE TABLE IF NOT EXISTS velib_forecast (
    identifiant_du_site_de_comptage BIGINT NOT NULL,
    date_et_heure_de_comptage TIMESTAMP WITHOUT TIME ZONE NOT NULL,
    comptage_horaire DOUBLE PRECISION NOT NULL,
    generated_at TIMESTAMP WITHOUT TIME ZONE NOT NULL DEFAULT NOW(),
    model_uri TEXT NULL,
    PRIMARY KEY (identifiant_du_site_de_comptage, date_et_heure_de_comptage)
);

-- 3) Optional supporting index to make time-based deletes and queries faster
CREATE INDEX IF NOT EXISTS idx_velib_forecast_datetime ON velib_forecast (date_et_heure_de_comptage);
CREATE INDEX IF NOT EXISTS idx_velib_forecast_site ON velib_forecast (identifiant_du_site_de_comptage);

-- 4) view to join forecast with site geometry for Streamlit heatmap
CREATE OR REPLACE VIEW velib_forecast_geo AS
SELECT
  f.identifiant_du_site_de_comptage,
  f.date_et_heure_de_comptage,
  f.comptage_horaire,
  s.nom_du_site_de_comptage,
  s.latitude,
  s.longitude,
  f.generated_at,
  f.model_uri
FROM velib_forecast f
LEFT JOIN velib_sites s
  USING (identifiant_du_site_de_comptage);