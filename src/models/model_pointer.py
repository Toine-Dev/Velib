from sqlalchemy import text
from sqlalchemy.engine import Engine
import json

def ensure_model_pointer_tables(engine: Engine) -> None:
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS model_pointer (
              name text PRIMARY KEY,
              run_id text NOT NULL,
              model_uri text NOT NULL,
              metrics jsonb,
              updated_at timestamptz NOT NULL DEFAULT now()
            );
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS model_pointer_history (
              id bigserial PRIMARY KEY,
              name text NOT NULL,
              run_id text NOT NULL,
              model_uri text NOT NULL,
              metrics jsonb,
              changed_at timestamptz NOT NULL DEFAULT now()
            );
        """))
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_model_pointer_history_name_time
            ON model_pointer_history (name, changed_at DESC);
        """))

def set_model_pointer(engine: Engine, name: str, run_id: str, model_uri: str, metrics: dict | None = None) -> None:
    ensure_model_pointer_tables(engine)
    metrics_json = None if metrics is None else json.dumps(metrics)

    with engine.begin() as conn:
        # 1) Append to history
        conn.execute(
            text("""
                INSERT INTO model_pointer_history (name, run_id, model_uri, metrics)
                VALUES (:name, :run_id, :model_uri, CAST(:metrics AS jsonb));
            """),
            {"name": name, "run_id": run_id, "model_uri": model_uri, "metrics": metrics_json},
        )

        # 2) Upsert current pointer
        conn.execute(
            text("""
                INSERT INTO model_pointer (name, run_id, model_uri, metrics, updated_at)
                VALUES (:name, :run_id, :model_uri, CAST(:metrics AS jsonb), now())
                ON CONFLICT (name) DO UPDATE SET
                    run_id = EXCLUDED.run_id,
                    model_uri = EXCLUDED.model_uri,
                    metrics = EXCLUDED.metrics,
                    updated_at = EXCLUDED.updated_at;
            """),
            {"name": name, "run_id": run_id, "model_uri": model_uri, "metrics": metrics_json},
        )

def get_model_pointer(engine: Engine, name: str = "production") -> dict:
    ensure_model_pointer_tables(engine)
    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT name, run_id, model_uri, metrics, updated_at FROM model_pointer WHERE name = :name"),
            {"name": name},
        ).mappings().first()
    if row is None:
        raise RuntimeError(f"No model pointer found for name={name!r}.")
    return dict(row)




# from sqlalchemy import text
# from sqlalchemy.engine import Engine

# def set_model_pointer(engine: Engine, name: str, run_id: str, model_uri: str, metrics: dict | None = None) -> None:
#     """
#     Save/update a pointer to the chosen model in Postgres.
#     """
#     with engine.begin() as conn:
#         conn.execute(text("""
#             CREATE TABLE IF NOT EXISTS model_pointer (
#               name text PRIMARY KEY,
#               run_id text NOT NULL,
#               model_uri text NOT NULL,
#               metrics jsonb,
#               created_at timestamptz NOT NULL DEFAULT now()
#             );
#         """))
#         conn.execute(
#             text("""
#                 INSERT INTO model_pointer (name, run_id, model_uri, metrics, created_at)
#                 VALUES (:name, :run_id, :model_uri, CAST(:metrics AS jsonb), now())
#                 ON CONFLICT (name) DO UPDATE SET
#                     run_id = EXCLUDED.run_id,
#                     model_uri = EXCLUDED.model_uri,
#                     metrics = EXCLUDED.metrics,
#                     created_at = EXCLUDED.created_at;
#             """),
#             {
#                 "name": name,
#                 "run_id": run_id,
#                 "model_uri": model_uri,
#                 "metrics": None if metrics is None else __import__("json").dumps(metrics),
#             },
#         )

# def get_model_pointer(engine: Engine, name: str = "production") -> dict:
#     with engine.connect() as conn:
#         row = conn.execute(
#             text("SELECT name, run_id, model_uri, metrics, created_at FROM model_pointer WHERE name = :name"),
#             {"name": name},
#         ).mappings().first()
#     if row is None:
#         raise RuntimeError(f"No model pointer found for name={name!r}.")
#     return dict(row)