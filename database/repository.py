"""Repository layer providing parameterized CRUD operations for PromptPilot."""

import json
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from database.db import DatabaseManager
from database.models import (
    DatasetRecord,
    EvaluationRecord,
    ExperimentRecord,
    FullExperimentDetail,
    PromptVersionRecord,
    ResponseRecord,
)
from utils.logging import get_logger

logger = get_logger("repository")


class ExperimentRepository:
    """Handles persistence of experiments, prompt versions, responses, and evaluations."""

    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager

    def create_experiment(
        self,
        task_type: str,
        original_prompt: str,
        input_text: str,
        model_name: str,
        title: Optional[str] = None,
        experiment_id: Optional[str] = None,
    ) -> str:
        """Create and persist a new experiment header."""
        exp_id = experiment_id or str(uuid.uuid4())
        created_at = datetime.now(timezone.utc).isoformat()
        auto_title = title or (original_prompt[:45] + "..." if len(original_prompt) > 45 else original_prompt)

        query = """
        INSERT INTO experiments (id, title, task_type, original_prompt, input_text, model_name, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?);
        """
        with self.db.get_connection() as conn:
            conn.execute(query, (exp_id, auto_title, task_type, original_prompt, input_text, model_name, created_at))

        logger.info("Created experiment %s (task: %s)", exp_id, task_type)
        return exp_id

    def add_prompt_version(
        self,
        experiment_id: str,
        version_number: int,
        strategy: str,
        prompt_text: str,
        expected_benefit: Optional[str] = None,
        potential_limitations: Optional[str] = None,
        version_id: Optional[str] = None,
    ) -> str:
        """Persist a prompt version associated with an experiment."""
        v_id = version_id or str(uuid.uuid4())
        created_at = datetime.now(timezone.utc).isoformat()

        query = """
        INSERT INTO prompt_versions (
            id, experiment_id, version_number, strategy, prompt_text,
            expected_benefit, potential_limitations, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?);
        """
        with self.db.get_connection() as conn:
            conn.execute(
                query,
                (v_id, experiment_id, version_number, strategy, prompt_text, expected_benefit, potential_limitations, created_at),
            )
        return v_id

    def add_response(
        self,
        prompt_version_id: str,
        response_text: str,
        latency_ms: float,
        prompt_tokens: Optional[int] = None,
        completion_tokens: Optional[int] = None,
        total_tokens: Optional[int] = None,
        response_id: Optional[str] = None,
    ) -> str:
        """Persist a response execution."""
        r_id = response_id or str(uuid.uuid4())
        created_at = datetime.now(timezone.utc).isoformat()

        query = """
        INSERT INTO responses (
            id, prompt_version_id, response_text, latency_ms,
            prompt_tokens, completion_tokens, total_tokens, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?);
        """
        with self.db.get_connection() as conn:
            conn.execute(
                query,
                (r_id, prompt_version_id, response_text, latency_ms, prompt_tokens, completion_tokens, total_tokens, created_at),
            )
        return r_id

    def add_evaluation(
        self,
        response_id: str,
        relevance: int,
        completeness: int,
        instruction_following: int,
        format_compliance: int,
        conciseness: int,
        factual_consistency: int,
        overall_score: float,
        feedback: List[str],
        strengths: List[str],
        evaluation_id: Optional[str] = None,
    ) -> str:
        """Persist evaluation scores and feedback for a response."""
        e_id = evaluation_id or str(uuid.uuid4())
        created_at = datetime.now(timezone.utc).isoformat()
        feedback_json = json.dumps(feedback)
        strengths_json = json.dumps(strengths)

        query = """
        INSERT INTO evaluations (
            id, response_id, relevance, completeness, instruction_following,
            format_compliance, conciseness, factual_consistency, overall_score,
            feedback, strengths, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """
        with self.db.get_connection() as conn:
            conn.execute(
                query,
                (
                    e_id, response_id, relevance, completeness, instruction_following,
                    format_compliance, conciseness, factual_consistency, overall_score,
                    feedback_json, strengths_json, created_at,
                ),
            )
        return e_id

    def get_experiment(self, experiment_id: str) -> Optional[FullExperimentDetail]:
        """Fetch complete hierarchical experiment tree (experiment, versions, responses, evaluations)."""
        with self.db.get_connection() as conn:
            # 1. Fetch experiment record
            exp_row = conn.execute(
                "SELECT id, title, task_type, original_prompt, input_text, model_name, created_at FROM experiments WHERE id = ?",
                (experiment_id,),
            ).fetchone()

            if not exp_row:
                return None

            experiment = ExperimentRecord(
                id=exp_row["id"],
                title=exp_row["title"],
                task_type=exp_row["task_type"],
                original_prompt=exp_row["original_prompt"],
                input_text=exp_row["input_text"],
                model_name=exp_row["model_name"],
                created_at=exp_row["created_at"],
            )

            # 2. Fetch all prompt versions for this experiment
            version_rows = conn.execute(
                """
                SELECT id, experiment_id, version_number, strategy, prompt_text,
                       expected_benefit, potential_limitations, created_at
                FROM prompt_versions
                WHERE experiment_id = ?
                ORDER BY version_number ASC
                """,
                (experiment_id,),
            ).fetchall()

            versions: List[PromptVersionRecord] = []
            for v_row in version_rows:
                v_record = PromptVersionRecord(
                    id=v_row["id"],
                    experiment_id=v_row["experiment_id"],
                    version_number=v_row["version_number"],
                    strategy=v_row["strategy"],
                    prompt_text=v_row["prompt_text"],
                    expected_benefit=v_row["expected_benefit"],
                    potential_limitations=v_row["potential_limitations"],
                    created_at=v_row["created_at"],
                )

                # 3. Fetch response for this version
                resp_row = conn.execute(
                    """
                    SELECT id, prompt_version_id, response_text, latency_ms,
                           prompt_tokens, completion_tokens, total_tokens, created_at
                    FROM responses
                    WHERE prompt_version_id = ?
                    ORDER BY created_at DESC
                    LIMIT 1
                    """,
                    (v_record.id,),
                ).fetchone()

                if resp_row:
                    resp_record = ResponseRecord(
                        id=resp_row["id"],
                        prompt_version_id=resp_row["prompt_version_id"],
                        response_text=resp_row["response_text"],
                        latency_ms=resp_row["latency_ms"],
                        prompt_tokens=resp_row["prompt_tokens"],
                        completion_tokens=resp_row["completion_tokens"],
                        total_tokens=resp_row["total_tokens"],
                        created_at=resp_row["created_at"],
                    )

                    # 4. Fetch evaluation for this response
                    eval_row = conn.execute(
                        """
                        SELECT id, response_id, relevance, completeness, instruction_following,
                               format_compliance, conciseness, factual_consistency, overall_score,
                               feedback, strengths, created_at
                        FROM evaluations
                        WHERE response_id = ?
                        """,
                        (resp_record.id,),
                    ).fetchone()

                    if eval_row:
                        resp_record.evaluation = EvaluationRecord(
                            id=eval_row["id"],
                            response_id=eval_row["response_id"],
                            relevance=eval_row["relevance"],
                            completeness=eval_row["completeness"],
                            instruction_following=eval_row["instruction_following"],
                            format_compliance=eval_row["format_compliance"],
                            conciseness=eval_row["conciseness"],
                            factual_consistency=eval_row["factual_consistency"],
                            overall_score=eval_row["overall_score"],
                            feedback=json.loads(eval_row["feedback"]) if eval_row["feedback"] else [],
                            strengths=json.loads(eval_row["strengths"]) if eval_row["strengths"] else [],
                            created_at=eval_row["created_at"],
                        )

                    v_record.response = resp_record

                versions.append(v_record)

            return FullExperimentDetail(experiment=experiment, versions=versions)

    def list_experiments(self, limit: int = 50, offset: int = 0) -> List[ExperimentRecord]:
        """Retrieve recent experiments for the history page."""
        query = """
        SELECT id, title, task_type, original_prompt, input_text, model_name, created_at
        FROM experiments
        ORDER BY created_at DESC
        LIMIT ? OFFSET ?;
        """
        with self.db.get_connection() as conn:
            rows = conn.execute(query, (limit, offset)).fetchall()
            return [
                ExperimentRecord(
                    id=r["id"],
                    title=r["title"],
                    task_type=r["task_type"],
                    original_prompt=r["original_prompt"],
                    input_text=r["input_text"],
                    model_name=r["model_name"],
                    created_at=r["created_at"],
                )
                for r in rows
            ]

    def delete_experiment(self, experiment_id: str) -> bool:
        """Delete an experiment and all cascaded children (versions, responses, evaluations)."""
        query = "DELETE FROM experiments WHERE id = ?;"
        with self.db.get_connection() as conn:
            cursor = conn.execute(query, (experiment_id,))
            return cursor.rowcount > 0

    def list_datasets(self) -> List[DatasetRecord]:
        """List all datasets with their test case counts."""
        query = """
        SELECT d.id, d.name, d.description, d.created_at, COUNT(c.id) AS case_count
        FROM datasets d
        LEFT JOIN dataset_cases c ON d.id = c.dataset_id
        GROUP BY d.id
        ORDER BY d.created_at ASC;
        """
        with self.db.get_connection() as conn:
            rows = conn.execute(query).fetchall()
            return [
                DatasetRecord(
                    id=r["id"],
                    name=r["name"],
                    description=r["description"] or "",
                    created_at=r["created_at"],
                    case_count=r["case_count"] or 0,
                )
                for r in rows
            ]

    def create_dataset(self, name: str, description: str = "", dataset_id: Optional[str] = None) -> str:
        """Create a new dataset container."""
        d_id = dataset_id or f"ds-{uuid.uuid4().hex[:8]}"
        now = datetime.now(timezone.utc).isoformat()
        query = """
        INSERT INTO datasets (id, name, description, created_at)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            name=excluded.name,
            description=excluded.description;
        """
        with self.db.get_connection() as conn:
            conn.execute(query, (d_id, name, description, now))
        return d_id

    def delete_dataset(self, dataset_id: str) -> bool:
        """Delete a dataset and its test cases (cascading)."""
        query = "DELETE FROM datasets WHERE id = ?;"
        with self.db.get_connection() as conn:
            cursor = conn.execute(query, (dataset_id,))
            return cursor.rowcount > 0

    def get_test_cases(self, dataset_id: str = "default-benchmark", task_type: Optional[str] = None) -> List[Any]:
        """Fetch all test cases belonging to a dataset."""
        from services.evaluation_service import TestCase
        query = "SELECT * FROM dataset_cases WHERE dataset_id = ?"
        params: List[Any] = [dataset_id]
        if task_type:
            query += " AND LOWER(task_type) = LOWER(?)"
            params.append(task_type)
        query += " ORDER BY created_at ASC;"
        with self.db.get_connection() as conn:
            rows = conn.execute(query, tuple(params)).fetchall()
            cases = []
            for r in rows:
                try:
                    tags = json.loads(r["tags"]) if r["tags"] else []
                except Exception:
                    tags = []
                cases.append(
                    TestCase(
                        id=r["id"],
                        dataset_id=r["dataset_id"],
                        name=r["name"],
                        task_type=r["task_type"],
                        description=r["description"] or "",
                        prompt=r["prompt"] or "",
                        input_text=r["input_text"] or "",
                        criteria=r["criteria"],
                        expected_output=r["expected_output"],
                        tags=tags,
                    )
                )
            return cases

    def get_test_case(self, case_id: str) -> Optional[Any]:
        """Retrieve a specific test case by ID."""
        from services.evaluation_service import TestCase
        query = "SELECT * FROM dataset_cases WHERE id = ?;"
        with self.db.get_connection() as conn:
            r = conn.execute(query, (case_id,)).fetchone()
            if not r:
                return None
            try:
                tags = json.loads(r["tags"]) if r["tags"] else []
            except Exception:
                tags = []
            return TestCase(
                id=r["id"],
                dataset_id=r["dataset_id"],
                name=r["name"],
                task_type=r["task_type"],
                description=r["description"] or "",
                prompt=r["prompt"] or "",
                input_text=r["input_text"] or "",
                criteria=r["criteria"],
                expected_output=r["expected_output"],
                tags=tags,
            )

    def save_test_case(self, case: Any) -> str:
        """Insert or update a test case in dataset_cases table."""
        c_id = getattr(case, "id", None) or getattr(case, "test_case_id", None) or f"tc-{uuid.uuid4().hex[:8]}"
        dataset_id = getattr(case, "dataset_id", "default-benchmark") or "default-benchmark"
        name = getattr(case, "name", "Untitled Test Case")
        task_type = getattr(case, "task_type", "Summarization")
        description = getattr(case, "description", "") or ""
        prompt = getattr(case, "prompt", "") or ""
        input_text = getattr(case, "input_text", "") or getattr(case, "source_data", "") or ""
        criteria = getattr(case, "criteria", None)
        expected_output = getattr(case, "expected_output", None)
        tags = getattr(case, "tags", [])
        tags_json = json.dumps(tags) if tags else "[]"
        now = datetime.now(timezone.utc).isoformat()

        with self.db.get_connection() as conn:
            ds_row = conn.execute("SELECT id FROM datasets WHERE id = ?", (dataset_id,)).fetchone()
            if not ds_row:
                conn.execute(
                    "INSERT INTO datasets (id, name, description, created_at) VALUES (?, ?, ?, ?)",
                    (dataset_id, dataset_id.replace("-", " ").title(), "User defined dataset", now),
                )

            query = """
            INSERT INTO dataset_cases (
                id, dataset_id, name, task_type, description, prompt, input_text,
                criteria, expected_output, tags, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                dataset_id=excluded.dataset_id,
                name=excluded.name,
                task_type=excluded.task_type,
                description=excluded.description,
                prompt=excluded.prompt,
                input_text=excluded.input_text,
                criteria=excluded.criteria,
                expected_output=excluded.expected_output,
                tags=excluded.tags;
            """
            conn.execute(
                query,
                (
                    c_id,
                    dataset_id,
                    name,
                    task_type,
                    description,
                    prompt,
                    input_text,
                    criteria,
                    expected_output,
                    tags_json,
                    now,
                ),
            )
        return c_id

    def delete_test_case(self, case_id: str) -> bool:
        """Delete an individual test case."""
        query = "DELETE FROM dataset_cases WHERE id = ?;"
        with self.db.get_connection() as conn:
            cursor = conn.execute(query, (case_id,))
            return cursor.rowcount > 0

    def save_case_result(self, result: Any) -> str:
        """Save an evaluated test case result."""
        res_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        ev = getattr(result, "evaluation", None)
        fb = json.dumps(ev.feedback) if ev and getattr(ev, "feedback", None) else "[]"
        st = json.dumps(ev.strengths) if ev and getattr(ev, "strengths", None) else "[]"

        query = """
        INSERT INTO dataset_case_results (
            id, test_case_id, dataset_id, prompt_text, source_data,
            response_text, latency_ms, overall_score, relevance,
            completeness, instruction_following, format_compliance,
            conciseness, factual_consistency, feedback, strengths, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """
        with self.db.get_connection() as conn:
            # Guarantee foreign key integrity
            c_check = conn.execute("SELECT id FROM dataset_cases WHERE id = ?", (result.test_case_id,)).fetchone()
            if not c_check:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO dataset_cases (
                        id, dataset_id, name, task_type, prompt, input_text, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (result.test_case_id, result.dataset_id, result.test_case_name, "General", result.prompt_text, result.source_data_preview, now),
                )

            conn.execute(
                query,
                (
                    res_id,
                    result.test_case_id,
                    result.dataset_id,
                    result.prompt_text,
                    result.source_data_preview,
                    result.response_text,
                    result.latency_ms,
                    ev.overall_score if ev else 0.0,
                    ev.relevance if ev else 0,
                    ev.completeness if ev else 0,
                    ev.instruction_following if ev else 0,
                    ev.effective_format_compliance if ev else 0,
                    ev.effective_conciseness if ev else 0,
                    ev.effective_factual_consistency if ev else 0,
                    fb,
                    st,
                    now,
                ),
            )
        return res_id

    def get_latest_case_results(self, dataset_id: Optional[str] = None, test_case_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieve recent case execution results."""
        query = """
        SELECT r.*, c.name as test_case_name
        FROM dataset_case_results r
        LEFT JOIN dataset_cases c ON r.test_case_id = c.id
        WHERE 1=1
        """
        params: List[Any] = []
        if dataset_id:
            query += " AND r.dataset_id = ?"
            params.append(dataset_id)
        if test_case_id:
            query += " AND r.test_case_id = ?"
            params.append(test_case_id)
        query += " ORDER BY r.created_at DESC LIMIT 50;"

        with self.db.get_connection() as conn:
            rows = conn.execute(query, tuple(params)).fetchall()
            return [dict(r) for r in rows]

