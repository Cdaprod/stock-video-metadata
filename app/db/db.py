# the Python interface module `/app/db/db.py` for application core
import sqlite3
from sqlite3 import Connection, Row
from typing import Any, Dict, List, Optional

DB_PATH = "app/db/schema.db"

def get_conn(db_path: str = DB_PATH) -> Connection:
    """Create and return a new SQLite connection with PRAGMAs set."""
    conn = sqlite3.connect(db_path, detect_types=sqlite3.PARSE_DECLTYPES)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA journal_mode = WAL;")
    return conn

class Database:
    """
    Database interface for CDAProd Video-Pipeline using SQLite.
    
    Provides methods to interact with batches, videos, attributes,
    enrichment runs, storage locations, events, and thumbnails.
    """
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path

    def _conn(self) -> Connection:
        return get_conn(self.db_path)

    # Batches
    def insert_batch(self, id: str, root_path: str, name: Optional[str] = None,
                     status: str = "new", notes: Optional[str] = None) -> None:
        sql = """
        INSERT INTO batches(id, root_path, name, status, notes)
        VALUES (?, ?, ?, ?, ?)
        """
        with self._conn() as conn:
            conn.execute(sql, (id, root_path, name, status, notes))

    def get_batch(self, id: str) -> Optional[Dict[str, Any]]:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM batches WHERE id = ?", (id,)).fetchone()
        return dict(row) if row else None

    # Videos
    def insert_video(self, id: str, batch_id: str, file_path: str, filename: str,
                     duration_s: Optional[float] = None, width_px: Optional[int] = None,
                     height_px: Optional[int] = None, fps: Optional[float] = None,
                     sha256: Optional[str] = None, current_state: str = "new") -> None:
        sql = """
        INSERT INTO videos(id, batch_id, file_path, filename,
                            duration_s, width_px, height_px, fps, sha256, current_state)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        with self._conn() as conn:
            conn.execute(sql, (id, batch_id, file_path, filename,
                               duration_s, width_px, height_px, fps, sha256, current_state))

    def get_video(self, id: str) -> Optional[Dict[str, Any]]:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM videos WHERE id = ?", (id,)).fetchone()
        return dict(row) if row else None

    def list_videos_by_batch(self, batch_id: str) -> List[Dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute("SELECT * FROM videos WHERE batch_id = ?", (batch_id,)).fetchall()
        return [dict(row) for row in rows]

    # Video Attributes
    def set_attribute(self, video_id: str, key: str, value: str) -> None:
        sql = """
        INSERT INTO video_attributes(video_id, key, value)
        VALUES (?, ?, ?)
        ON CONFLICT(video_id, key) DO UPDATE SET value = excluded.value, updated_at = CURRENT_TIMESTAMP
        """
        with self._conn() as conn:
            conn.execute(sql, (video_id, key, value))

    def get_attributes(self, video_id: str) -> Dict[str, str]:
        with self._conn() as conn:
            rows = conn.execute("SELECT key, value FROM video_attributes WHERE video_id = ?", (video_id,)).fetchall()
        return {row["key"]: row["value"] for row in rows}

    # Enrichment Runs
    def create_enrichment_run(self, video_id: str, step_name: str,
                              step_version: Optional[str] = None,
                              parameters: Optional[str] = None) -> int:
        sql = """
        INSERT INTO enrichment_runs(video_id, step_name, step_version, parameters)
        VALUES (?, ?, ?, ?)
        """
        with self._conn() as conn:
            cursor = conn.execute(sql, (video_id, step_name, step_version, parameters))
            return cursor.lastrowid

    def update_enrichment_run(self, run_id: int, status: str,
                              output: Optional[str] = None) -> None:
        sql = """
        UPDATE enrichment_runs
           SET status = ?, output = ?, ended_at = CURRENT_TIMESTAMP
         WHERE run_id = ?
        """
        with self._conn() as conn:
            conn.execute(sql, (status, output, run_id))

    def list_enrichment_runs(self, video_id: str) -> List[Dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM enrichment_runs WHERE video_id = ? ORDER BY started_at",
                (video_id,)
            ).fetchall()
        return [dict(row) for row in rows]

    # Storage Locations
    def add_storage_location(self, video_id: str, storage_tier: str, uri: str) -> None:
        sql = """
        INSERT INTO storage_locations(video_id, storage_tier, uri)
        VALUES (?, ?, ?)
        """
        with self._conn() as conn:
            conn.execute(sql, (video_id, storage_tier, uri))

    def get_storage_locations(self, video_id: str) -> List[Dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM storage_locations WHERE video_id = ?",
                (video_id,)
            ).fetchall()
        return [dict(row) for row in rows]

    # Events
    def log_event(self, aggregate: str, aggregate_id: str,
                  event_type: str, event_data: str) -> int:
        sql = """
        INSERT INTO events(aggregate, aggregate_id, event_type, event_data)
        VALUES (?, ?, ?, ?)
        """
        with self._conn() as conn:
            cursor = conn.execute(sql, (aggregate, aggregate_id, event_type, event_data))
            return cursor.lastrowid

    def get_events(self, aggregate: str, aggregate_id: Optional[str] = None) -> List[Dict[str, Any]]:
        if aggregate_id:
            sql = "SELECT * FROM events WHERE aggregate = ? AND aggregate_id = ? ORDER BY created_at"
            params = (aggregate, aggregate_id)
        else:
            sql = "SELECT * FROM events WHERE aggregate = ? ORDER BY created_at"
            params = (aggregate,)
        with self._conn() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [dict(row) for row in rows]

    # Thumbnails
    def add_thumbnail(self, video_id: str, kind: str, file_path: str,
                      width_px: Optional[int] = None, height_px: Optional[int] = None) -> None:
        sql = """
        INSERT INTO thumbnails(video_id, kind, file_path, width_px, height_px)
        VALUES (?, ?, ?, ?, ?)
        """
        with self._conn() as conn:
            conn.execute(sql, (video_id, kind, file_path, width_px, height_px))

    def get_thumbnails(self, video_id: str) -> List[Dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM thumbnails WHERE video_id = ? ORDER BY created_at",
                (video_id,)
            ).fetchall()
        return [dict(row) for row in rows]

# Display the constructed file content to the user
import pandas as pd
from ace_tools import display_dataframe_to_user

# Show a snippet of method names for quick reference
methods = [func for func in dir(Database) if not func.startswith("_")]
df = pd.DataFrame({"Database Methods": methods})
display_dataframe_to_user("app/db.py Interface Methods", df)