"""SQLite database helpers for the leaderboard."""

import sqlite3
from pathlib import Path

DEFAULT_DB_PATH = Path(__file__).parent / "leaderboard.db"


def init_db(db_path: Path = DEFAULT_DB_PATH) -> None:
    """Create the submissions table if it doesn't exist."""
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS submissions (
            team_name TEXT NOT NULL,
            resume_id TEXT NOT NULL,
            score REAL NOT NULL,
            submitted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(team_name, resume_id)
        )
        """
    )
    conn.commit()
    conn.close()


def add_submission(
    db_path: Path, team_name: str, resume_id: str, score: float
) -> None:
    """Insert or replace a submission (teams can update scores)."""
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        INSERT OR REPLACE INTO submissions (team_name, resume_id, score, submitted_at)
        VALUES (?, ?, ?, CURRENT_TIMESTAMP)
        """,
        (team_name, resume_id, score),
    )
    conn.commit()
    conn.close()


def get_all_submissions(db_path: Path) -> list[dict]:
    """Return all submissions as a list of dicts."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT team_name, resume_id, score, submitted_at FROM submissions "
        "ORDER BY submitted_at DESC"
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def reset_db(db_path: Path) -> None:
    """Delete all submissions."""
    conn = sqlite3.connect(db_path)
    conn.execute("DELETE FROM submissions")
    conn.commit()
    conn.close()
