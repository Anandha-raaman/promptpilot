"""SQLite database initialization and connection management for PromptPilot.

Ensures strict relational integrity using foreign keys, parameterized executions,
and connection pooling context managers.
"""

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Optional

from config.settings import Settings, get_settings
from utils.logging import get_logger

logger = get_logger("database")

SCHEMA_SQL = """
-- Core Experiments Table
CREATE TABLE IF NOT EXISTS experiments (
    id TEXT PRIMARY KEY,
    title TEXT,
    task_type TEXT NOT NULL,
    original_prompt TEXT NOT NULL,
    input_text TEXT NOT NULL,
    model_name TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_experiments_created ON experiments(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_experiments_task ON experiments(task_type);

-- Prompt Variants / Iterations Table
CREATE TABLE IF NOT EXISTS prompt_versions (
    id TEXT PRIMARY KEY,
    experiment_id TEXT NOT NULL,
    version_number INTEGER NOT NULL,
    strategy TEXT NOT NULL,
    prompt_text TEXT NOT NULL,
    expected_benefit TEXT,
    potential_limitations TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (experiment_id) REFERENCES experiments(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_versions_experiment ON prompt_versions(experiment_id, version_number);

-- Responses Table
CREATE TABLE IF NOT EXISTS responses (
    id TEXT PRIMARY KEY,
    prompt_version_id TEXT NOT NULL,
    response_text TEXT NOT NULL,
    latency_ms REAL NOT NULL,
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    total_tokens INTEGER,
    created_at TEXT NOT NULL,
    FOREIGN KEY (prompt_version_id) REFERENCES prompt_versions(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_responses_version ON responses(prompt_version_id);

-- Evaluation Scores Table
CREATE TABLE IF NOT EXISTS evaluations (
    id TEXT PRIMARY KEY,
    response_id TEXT NOT NULL UNIQUE,
    relevance INTEGER NOT NULL,
    completeness INTEGER NOT NULL,
    instruction_following INTEGER NOT NULL,
    format_compliance INTEGER NOT NULL,
    conciseness INTEGER NOT NULL,
    factual_consistency INTEGER NOT NULL,
    overall_score REAL NOT NULL,
    feedback TEXT NOT NULL, -- JSON-encoded string
    strengths TEXT NOT NULL, -- JSON-encoded string
    created_at TEXT NOT NULL,
    FOREIGN KEY (response_id) REFERENCES responses(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_evaluations_response ON evaluations(response_id);

-- Datasets Table
CREATE TABLE IF NOT EXISTS datasets (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    created_at TEXT NOT NULL
);

-- Dataset Test Cases Table
CREATE TABLE IF NOT EXISTS dataset_cases (
    id TEXT PRIMARY KEY,
    dataset_id TEXT NOT NULL,
    name TEXT NOT NULL,
    task_type TEXT NOT NULL,
    description TEXT,
    prompt TEXT NOT NULL,
    input_text TEXT NOT NULL,
    criteria TEXT,
    expected_output TEXT,
    tags TEXT, -- JSON-encoded list of tags
    created_at TEXT NOT NULL,
    FOREIGN KEY (dataset_id) REFERENCES datasets(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_cases_dataset ON dataset_cases(dataset_id);
CREATE INDEX IF NOT EXISTS idx_cases_task ON dataset_cases(task_type);

-- Dataset Case Results Table
CREATE TABLE IF NOT EXISTS dataset_case_results (
    id TEXT PRIMARY KEY,
    test_case_id TEXT NOT NULL,
    dataset_id TEXT NOT NULL,
    prompt_text TEXT NOT NULL,
    source_data TEXT NOT NULL,
    response_text TEXT NOT NULL,
    latency_ms REAL NOT NULL,
    overall_score REAL NOT NULL,
    relevance INTEGER NOT NULL,
    completeness INTEGER NOT NULL,
    instruction_following INTEGER NOT NULL,
    format_compliance INTEGER NOT NULL,
    conciseness INTEGER NOT NULL,
    factual_consistency INTEGER NOT NULL,
    feedback TEXT NOT NULL, -- JSON-encoded
    strengths TEXT NOT NULL, -- JSON-encoded
    created_at TEXT NOT NULL,
    FOREIGN KEY (test_case_id) REFERENCES dataset_cases(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_results_case ON dataset_case_results(test_case_id);
CREATE INDEX IF NOT EXISTS idx_results_dataset ON dataset_case_results(dataset_id);
"""


class DatabaseManager:
    """Manages SQLite connections and schema initialization."""

    def __init__(self, db_path: Optional[Path] = None, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self.db_path = db_path or self.settings.get_resolved_db_path()
        self._init_db()

    def _init_db(self) -> None:
        """Apply schema and create database file if not present."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self.get_connection() as conn:
            conn.executescript(SCHEMA_SQL)
        self._seed_default_dataset()
        logger.info("Database initialized successfully at: %s", self.db_path)

    def _seed_default_dataset(self) -> None:
        """Seed default benchmark dataset and test cases from test_cases.json if empty."""
        import json
        from datetime import datetime, timezone

        try:
            with self.get_connection() as conn:
                row = conn.execute("SELECT COUNT(*) AS cnt FROM datasets WHERE id = 'default-benchmark'").fetchone()
                if row and row["cnt"] == 0:
                    now = datetime.now(timezone.utc).isoformat()
                    conn.execute(
                        "INSERT INTO datasets (id, name, description, created_at) VALUES (?, ?, ?, ?)",
                        ("default-benchmark", "Default Benchmark Dataset", "Standard benchmark suite covering Summarization, Information Extraction, and Text Generation.", now),
                    )

                case_row = conn.execute("SELECT COUNT(*) AS cnt FROM dataset_cases WHERE dataset_id = 'default-benchmark'").fetchone()
                if case_row and case_row["cnt"] == 0:
                    json_path = Path(__file__).resolve().parent.parent / "data" / "test_cases.json"
                    if json_path.exists():
                        with open(json_path, "r", encoding="utf-8") as f:
                            cases = json.load(f)
                        now = datetime.now(timezone.utc).isoformat()
                        for c in cases:
                            conn.execute(
                                """
                                INSERT OR IGNORE INTO dataset_cases (
                                    id, dataset_id, name, task_type, description,
                                    prompt, input_text, criteria, expected_output, tags, created_at
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
        except Exception as e:
            logger.warning("Failed to seed default benchmark dataset: %s", e)

    @contextmanager
    def get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Provide a contextual transactional SQLite connection with foreign keys enabled."""
        conn = sqlite3.connect(
            database=str(self.db_path),
            timeout=10.0,
            check_same_thread=False,
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

