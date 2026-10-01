"""Sync dataset_cases in SQLite database with data/test_cases.json."""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


def sync():
    db_path = Path("data/promptpilot.db")
    json_path = Path("data/test_cases.json")
    with open(json_path, "r", encoding="utf-8") as f:
        cases = json.load(f)

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Clear existing cases in default-benchmark
    cur.execute("DELETE FROM dataset_cases WHERE dataset_id = 'default-benchmark'")

    now = datetime.now(timezone.utc).isoformat()
    for c in cases:
        cur.execute(
            """
            INSERT INTO dataset_cases (
                id, dataset_id, name, task_type, description, prompt, input_text,
                criteria, expected_output, tags, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                c["id"],
                c.get("dataset_id", "default-benchmark"),
                c["name"],
                c["task_type"],
                c.get("description", ""),
                c.get("prompt", ""),
                c["input_text"],
                c.get("criteria", ""),
                c.get("expected_output", ""),
                json.dumps(c.get("tags", [])),
                now,
            ),
        )
    conn.commit()

    rows = cur.execute("SELECT id, name, task_type FROM dataset_cases WHERE dataset_id = 'default-benchmark'").fetchall()
    print(f"Successfully synced {len(rows)} cases to default-benchmark:")
    for r in rows:
        print(f" - {r[0]}: {r[1]} ({r[2]})")
    conn.close()


if __name__ == "__main__":
    sync()
