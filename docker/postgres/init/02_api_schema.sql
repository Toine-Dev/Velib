-- Create dedicated schema for API objects
CREATE SCHEMA IF NOT EXISTS api;

-- Users table
CREATE TABLE IF NOT EXISTS api.users (
    username TEXT PRIMARY KEY,
    password_hash TEXT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('admin','client')),
    created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT NOW()
);

-- Pipeline job queue
CREATE TABLE IF NOT EXISTS api.pipeline_jobs (
    id BIGSERIAL PRIMARY KEY,
    job_type TEXT NOT NULL CHECK (job_type IN ('data','model','forecast')),
    status TEXT NOT NULL DEFAULT 'queued'
        CHECK (status IN ('queued','running','success','failed')),
    created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT NOW(),
    started_at TIMESTAMP WITHOUT TIME ZONE,
    finished_at TIMESTAMP WITHOUT TIME ZONE,
    message TEXT
);

-- -- Default admin
-- INSERT INTO api.users(username,password_hash,role)
-- VALUES ('admin','$2b$12$1wI6yNqXfZVYJ2lq1m0QjO3g6k7QXkJqf7h5gH9K5o3cQ0d2Hq7yK','admin')
-- ON CONFLICT (username) DO NOTHING;