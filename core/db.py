import os
import sqlite3

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "traffic.db")


def get_connection():
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=30000")
    return conn


def init_db():
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_name TEXT NOT NULL,
                motorbike INTEGER NOT NULL DEFAULT 0,
                car INTEGER NOT NULL DEFAULT 0,
                bus INTEGER NOT NULL DEFAULT 0,
                truck INTEGER NOT NULL DEFAULT 0,
                traffic_status TEXT NOT NULL DEFAULT 'Binh thuong',
                auto_comment TEXT NOT NULL DEFAULT '',
                peak_minute INTEGER,
                peak_flow INTEGER NOT NULL DEFAULT 0,
                avg_speed REAL NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL
            )
            """
        )
        cursor.execute("PRAGMA table_info(results)")
        existing_columns = {row[1] for row in cursor.fetchall()}

        alter_statements = [
            ("traffic_status", "ALTER TABLE results ADD COLUMN traffic_status TEXT NOT NULL DEFAULT 'Binh thuong'"),
            ("auto_comment", "ALTER TABLE results ADD COLUMN auto_comment TEXT NOT NULL DEFAULT ''"),
            ("peak_minute", "ALTER TABLE results ADD COLUMN peak_minute INTEGER"),
            ("peak_flow", "ALTER TABLE results ADD COLUMN peak_flow INTEGER NOT NULL DEFAULT 0"),
            ("avg_speed", "ALTER TABLE results ADD COLUMN avg_speed REAL NOT NULL DEFAULT 0"),
        ]

        for column_name, statement in alter_statements:
            if column_name not in existing_columns:
                cursor.execute(statement)

        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_results_created_at
            ON results(created_at)
            """
        )
