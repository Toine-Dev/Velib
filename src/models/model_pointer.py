from sqlalchemy import text
from sqlalchemy.engine import Engine

def set_model_pointer(engine: Engine, name: str, run_id: str, model_uri: str, metrics: dict | None = None) -> None:
    """
    Save/update a pointer to the chosen model in Postgres.
    """
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS model_pointer (
              name text PRIMARY KEY,
              run_id text NOT NULL,
              model_uri text NOT NULL,
              metrics jsonb,
              created_at timestamptz NOT NULL DEFAULT now()
            );
        """))
        conn.execute(
            text("""
                INSERT INTO model_pointer (name, run_id, model_uri, metrics, created_at)
                VALUES (:name, :run_id, :model_uri, CAST(:metrics AS jsonb), now())
                ON CONFLICT (name) DO UPDATE SET
                    run_id = EXCLUDED.run_id,
                    model_uri = EXCLUDED.model_uri,
                    metrics = EXCLUDED.metrics,
                    created_at = EXCLUDED.created_at;
            """),
            {
                "name": name,
                "run_id": run_id,
                "model_uri": model_uri,
                "metrics": None if metrics is None else __import__("json").dumps(metrics),
            },
        )

def get_model_pointer(engine: Engine, name: str = "production") -> dict:
    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT name, run_id, model_uri, metrics, created_at FROM model_pointer WHERE name = :name"),
            {"name": name},
        ).mappings().first()
    if row is None:
        raise RuntimeError(f"No model pointer found for name={name!r}.")
    return dict(row)