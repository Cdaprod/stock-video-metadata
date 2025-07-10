-- ============================================================================
-- CDAProd • Video-Pipeline Database Schema
-- ============================================================================

PRAGMA journal_mode=WAL;          -- concurrent reads, safe writes
PRAGMA foreign_keys = ON;
PRAGMA busy_timeout = 10000;      -- 10 s for lock contention
PRAGMA user_version = 1;          -- bump when you migrate

/* ---------------------------------------------------------------------------
   1.  Batches & Videos
   -------------------------------------------------------------------------*/
CREATE TABLE IF NOT EXISTS batches (
    id            TEXT PRIMARY KEY,          -- e.g. UUID or slug
    root_path     TEXT NOT NULL,             -- /mnt/batches/Batch-42
    name          TEXT,                      -- optional friendly label
    discovered_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    status        TEXT DEFAULT 'new',        -- new|processing|done|error
    notes         TEXT
);

CREATE TABLE IF NOT EXISTS videos (
    id            TEXT PRIMARY KEY,          -- UUID / SHA-256 of file
    batch_id      TEXT NOT NULL,
    file_path     TEXT NOT NULL,             -- absolute on ingest
    filename      TEXT NOT NULL,
    duration_s    REAL,                      -- seconds (nullable until probed)
    width_px      INTEGER,
    height_px     INTEGER,
    fps           REAL,
    sha256        TEXT UNIQUE,               -- if id is not the hash itself
    discovered_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    current_state TEXT NOT NULL DEFAULT 'new',   -- new|enriching|ready|archived
    FOREIGN KEY(batch_id) REFERENCES batches(id) ON DELETE CASCADE
);

/* ---------------------------------------------------------------------------
   2.  Flexible Key/Value Attributes  (one-to-many, evolvable)
   -------------------------------------------------------------------------*/
CREATE TABLE IF NOT EXISTS video_attributes (
    attr_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id      TEXT NOT NULL,
    key           TEXT NOT NULL,             -- e.g. 'yolo_objects', 'blip_caption'
    value         TEXT NOT NULL,             -- JSON or plain text
    updated_at    TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(video_id, key),                   -- last write wins
    FOREIGN KEY(video_id) REFERENCES videos(id) ON DELETE CASCADE
);

/* ---------------------------------------------------------------------------
   3.  Enrichment Runs  (pluggable step log & output blob)
   -------------------------------------------------------------------------*/
CREATE TABLE IF NOT EXISTS enrichment_runs (
    run_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id      TEXT NOT NULL,
    step_name     TEXT NOT NULL,             -- 'blip', 'yolo', 'ocr', ...
    step_version  TEXT,                      -- git-sha / docker tag
    parameters    TEXT,                      -- JSON-encoded params/input
    output        TEXT,                      -- JSON-encoded result
    status        TEXT NOT NULL DEFAULT 'started', -- started|done|error
    started_at    TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    ended_at      TIMESTAMP,
    FOREIGN KEY(video_id) REFERENCES videos(id) ON DELETE CASCADE
);

/* ---------------------------------------------------------------------------
   4.  Storage Lifecycle  (object-store tiers, offloads, purge history)
   -------------------------------------------------------------------------*/
CREATE TABLE IF NOT EXISTS storage_locations (
    loc_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id      TEXT NOT NULL,
    storage_tier  TEXT NOT NULL,             -- local|hot|cold|archive
    uri           TEXT NOT NULL,             -- s3://bucket/key  or  file://…
    version       INTEGER NOT NULL DEFAULT 1,
    stored_at     TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    removed_at    TIMESTAMP,
    FOREIGN KEY(video_id) REFERENCES videos(id) ON DELETE CASCADE
);

/* ---------------------------------------------------------------------------
   5.  Event Sourcing  (immutable audit/history for everything)
   -------------------------------------------------------------------------*/
CREATE TABLE IF NOT EXISTS events (
    event_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    aggregate     TEXT NOT NULL,             -- 'video' | 'batch' | 'system'
    aggregate_id  TEXT NOT NULL,
    event_type    TEXT NOT NULL,             -- 'STATE_CHANGED', 'ERROR', …
    event_data    TEXT NOT NULL,             -- JSON body
    created_at    TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

/* ---------------------------------------------------------------------------
   6.  Thumbnails / Derivative Artifacts
   -------------------------------------------------------------------------*/
CREATE TABLE IF NOT EXISTS thumbnails (
    thumb_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id      TEXT NOT NULL,
    kind          TEXT NOT NULL,             -- 'main' | 'frame-n' | 'contact-sheet'
    file_path     TEXT NOT NULL,
    width_px      INTEGER,
    height_px     INTEGER,
    created_at    TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(video_id) REFERENCES videos(id) ON DELETE CASCADE
);

/* ---------------------------------------------------------------------------
   7.  Indices for Hot Paths
   -------------------------------------------------------------------------*/
CREATE INDEX IF NOT EXISTS idx_videos_batch      ON videos(batch_id);
CREATE INDEX IF NOT EXISTS idx_attr_video_key    ON video_attributes(video_id, key);
CREATE INDEX IF NOT EXISTS idx_enrich_video_step ON enrichment_runs(video_id, step_name);
CREATE INDEX IF NOT EXISTS idx_storage_video     ON storage_locations(video_id);
CREATE INDEX IF NOT EXISTS idx_events_agg        ON events(aggregate, aggregate_id);

/* ---------------------------------------------------------------------------
   8.  Convenience Views
   -------------------------------------------------------------------------*/
-- Latest attribute key/value flattened per video (handy for dashboards)
CREATE VIEW IF NOT EXISTS v_video_meta AS
SELECT
    v.id             AS video_id,
    v.filename,
    v.duration_s,
    v.width_px,
    v.height_px,
    v.fps,
    v.current_state,
    json_group_object(a.key, a.value) AS meta_json
FROM videos v
LEFT JOIN video_attributes a
      ON a.video_id = v.id
GROUP BY v.id;

-- Completed enrichment run counts per step
CREATE VIEW IF NOT EXISTS v_enrichment_stats AS
SELECT
    step_name,
    COUNT(*)                AS run_count,
    SUM(status = 'error')   AS error_count,
    ROUND(AVG(julianday(ended_at) - julianday(started_at)) * 86400, 2) AS avg_seconds
FROM enrichment_runs
WHERE status = 'done'
GROUP BY step_name;

/*
-- Create the DB once
sqlite3 ~/data/metadata.db < /db/schema.sql
-- Inspect
sqlite3 ~/data/metadata.db ".tables" 
*\