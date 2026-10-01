"""Sanitized export utilities for experiments, evaluations, and comparison reports.

Generates JSON, CSV, and Markdown formats while strictly excluding credentials,
tokens, or environment secrets.
"""

import csv
import io
import json
from datetime import datetime, timezone
from typing import Any, Dict, List
from database.models import FullExperimentDetail
from utils.logging import redact_secrets


class ExperimentExporter:
    """Exports experiment datasets in multiple clean formats."""

    @staticmethod
    def to_json(detail: FullExperimentDetail, indent: int = 2) -> str:
        """Export experiment tree to sanitized JSON."""
        data = {
            "promptpilot_version": "1.0.0",
            "export_timestamp": datetime.now(timezone.utc).isoformat(),
            "experiment": {
                "id": detail.experiment.id,
                "title": detail.experiment.title,
                "task_type": detail.experiment.task_type,
                "original_prompt": detail.experiment.original_prompt,
                "input_text": detail.experiment.input_text,
                "model_name": detail.experiment.model_name,
                "created_at": detail.experiment.created_at,
            },
            "versions": [],
        }

        for v in detail.versions:
            v_dict: Dict[str, Any] = {
                "version_number": v.version_number,
                "strategy": v.strategy,
                "prompt_text": v.prompt_text,
                "expected_benefit": v.expected_benefit,
                "potential_limitations": v.potential_limitations,
                "response": None,
            }
            if v.response:
                v_dict["response"] = {
                    "text": v.response.response_text,
                    "latency_ms": v.response.latency_ms,
                    "prompt_tokens": v.response.prompt_tokens,
                    "completion_tokens": v.response.completion_tokens,
                    "total_tokens": v.response.total_tokens,
                    "evaluation": None,
                }
                if v.response.evaluation:
                    ev = v.response.evaluation
                    v_dict["response"]["evaluation"] = {
                        "relevance": ev.relevance,
                        "completeness": ev.completeness,
                        "instruction_following": ev.instruction_following,
                        "format_compliance": ev.format_compliance,
                        "conciseness": ev.conciseness,
                        "factual_consistency": ev.factual_consistency,
                        "overall_score": ev.overall_score,
                        "feedback": ev.feedback,
                        "strengths": ev.strengths,
                    }
            data["versions"].append(v_dict)

        return redact_secrets(json.dumps(data, indent=indent))

    @staticmethod
    def to_csv(detail: FullExperimentDetail) -> str:
        """Export summary comparison metrics to CSV format."""
        output = io.StringIO()
        fieldnames = [
            "experiment_id",
            "task_type",
            "version_number",
            "strategy",
            "prompt_text",
            "latency_ms",
            "relevance",
            "completeness",
            "instruction_following",
            "format_compliance",
            "conciseness",
            "factual_consistency",
            "overall_score",
            "response_snippet",
        ]
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()

        for v in detail.versions:
            row = {
                "experiment_id": detail.experiment.id,
                "task_type": detail.experiment.task_type,
                "version_number": v.version_number,
                "strategy": v.strategy,
                "prompt_text": v.prompt_text.replace("\n", " "),
                "latency_ms": v.response.latency_ms if v.response else 0.0,
                "relevance": v.response.evaluation.relevance if (v.response and v.response.evaluation) else "",
                "completeness": v.response.evaluation.completeness if (v.response and v.response.evaluation) else "",
                "instruction_following": v.response.evaluation.instruction_following if (v.response and v.response.evaluation) else "",
                "format_compliance": v.response.evaluation.format_compliance if (v.response and v.response.evaluation) else "",
                "conciseness": v.response.evaluation.conciseness if (v.response and v.response.evaluation) else "",
                "factual_consistency": v.response.evaluation.factual_consistency if (v.response and v.response.evaluation) else "",
                "overall_score": v.response.evaluation.overall_score if (v.response and v.response.evaluation) else "",
                "response_snippet": (v.response.response_text[:80] + "...") if v.response else "",
            }
            writer.writerow(row)

        return redact_secrets(output.getvalue())

    @staticmethod
    def to_markdown(detail: FullExperimentDetail) -> str:
        """Generate a complete Markdown audit report."""
        lines = [
            f"# PromptPilot Experiment Report: {detail.experiment.title or detail.experiment.id}",
            f"- **Date**: {detail.experiment.created_at}",
            f"- **Task Type**: {detail.experiment.task_type}",
            f"- **Model**: `{detail.experiment.model_name}`",
            "",
            "## 1. Source Context & Baseline Prompt",
            "### Original Prompt",
            "```text",
            detail.experiment.original_prompt,
            "```",
            "",
            "### Source Data Input",
            "```text",
            detail.experiment.input_text[:500] + ("..." if len(detail.experiment.input_text) > 500 else ""),
            "```",
            "",
            "## 2. Prompt Versions & Model-Based Evaluation Summary",
            "| Version | Strategy | Overall Score (0-10) | Latency (ms) | Tokens |",
            "| :--- | :--- | :--- | :--- | :--- |",
        ]

        for v in detail.versions:
            score = v.response.evaluation.overall_score if (v.response and v.response.evaluation) else "N/A"
            latency = f"{v.response.latency_ms:.1f}" if v.response else "N/A"
            tokens = v.response.total_tokens if (v.response and v.response.total_tokens) else "N/A"
            lines.append(f"| V{v.version_number} | {v.strategy} | **{score}** | {latency} | {tokens} |")

        lines.extend([
            "",
            "## 3. Detailed Version Findings",
        ])

        for v in detail.versions:
            lines.extend([
                f"### Version {v.version_number}: {v.strategy}",
                f"- **Expected Benefit**: {v.expected_benefit or 'N/A'}",
                f"- **Potential Limitations**: {v.potential_limitations or 'N/A'}",
                "",
                "**Prompt Text**:",
                "```text",
                v.prompt_text,
                "```",
            ])
            if v.response:
                lines.extend([
                    "",
                    "**Generated Completion**:",
                    "> " + v.response.response_text.replace("\n", "\n> "),
                ])
                if v.response.evaluation:
                    ev = v.response.evaluation
                    lines.extend([
                        "",
                        f"**Evaluation Score**: `{ev.overall_score} / 10.0`",
                        f"- Relevance: {ev.relevance}/10 | Completeness: {ev.completeness}/10 | Instruction Following: {ev.instruction_following}/10",
                        f"- Format: {ev.format_compliance}/10 | Conciseness: {ev.conciseness}/10 | Factual Consistency: {ev.factual_consistency}/10",
                    ])
                    if ev.feedback:
                        lines.append("**Feedback**:")
                        for fb in ev.feedback:
                            lines.append(f"  * {fb}")
                    if ev.strengths:
                        lines.append("**Strengths**:")
                        for st in ev.strengths:
                            lines.append(f"  * {st}")
            lines.append("")

        lines.extend([
            "---",
            "*Report generated by PromptPilot AI Optimization Workbench.*",
            "*Note: Model-based evaluation is experimental and should be verified with human-in-the-loop review.*",
        ])

        return redact_secrets("\n".join(lines))
