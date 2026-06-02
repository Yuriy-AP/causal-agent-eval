"""LLM-as-judge evaluation logger for CausalAgentBench agent runs."""

from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from openinference.instrumentation import suppress_tracing
from openinference.semconv.trace import SpanAttributes
from phoenix.client.types.spans import SpanQuery
from phoenix.evals import LLM, create_classifier

from agents.financial_analyst_agent import PROJECT_NAME

DEFAULT_LOG_PATH = Path("experiments/run_logs.csv")

GROUNDEDNESS_TEMPLATE = """
You are an expert evaluator of financial analysis summaries.

Determine whether the summary is factually grounded in the provided metrics.
The summary must not invent numbers, misstate metrics, or claim relationships unsupported by the data.

## Metrics (source of truth)
{metrics}

## Agent summary
{summary}

Classify with exactly one label: grounded or not_grounded.
"""


def _normalize_groundedness_label(label: str) -> str:
    """Map judge output to valid rails when the model returns prose instead of a label."""
    normalized = label.strip().lower()
    if normalized in {"grounded", "not_grounded"}:
        return normalized
    if "not_grounded" in normalized or normalized.startswith("not "):
        return "not_grounded"
    if any(token in normalized for token in ("unsupported", "incorrect", "invented", "misstate")):
        return "not_grounded"
    if "grounded" in normalized or "align" in normalized:
        return "grounded"
    return "not_grounded"


def _build_groundedness_evaluator(judge_model: str):
    llm = LLM(provider="openai", model=judge_model)
    return create_classifier(
        name="summary_groundedness",
        prompt_template=GROUNDEDNESS_TEMPLATE,
        llm=llm,
        choices={
            "grounded": 1.0,
            "not_grounded": 0.0,
        },
        direction="maximize",
    )


def score_summary_groundedness(
    summary: str,
    metrics: dict[str, Any],
    *,
    judge_model: str = "gpt-4o-mini",
) -> tuple[float, str]:
    """
    Use phoenix.evals LLM-as-judge to score whether the summary is grounded in metrics.

    Returns (score 0-1, explanation).
    """
    if not summary.strip():
        return 0.0, "Empty summary."

    if not metrics:
        return 0.0, "No metrics available to verify against."

    with suppress_tracing():
        evaluator = _build_groundedness_evaluator(judge_model)
        try:
            scores = evaluator.evaluate(
                {
                    "summary": summary,
                    "metrics": json.dumps(metrics, indent=2),
                }
            )
            result = scores[0]
            label = _normalize_groundedness_label(result.label or "")
            score = float(
                result.score
                if result.score is not None
                else (1.0 if label == "grounded" else 0.0)
            )
            return score, result.explanation or ""
        except ValueError as exc:
            if "invalid label" not in str(exc):
                raise
            match = re.search(r"invalid label '([^']+)'", str(exc))
            raw_label = match.group(1) if match else ""
            label = _normalize_groundedness_label(raw_label)
            score = 1.0 if label == "grounded" else 0.0
            return score, raw_label


def _parse_summary_output(value: Any) -> str | None:
    """Normalize tool span output to plain summary text."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict) and parsed.get("summary"):
                return str(parsed["summary"])
        except json.JSONDecodeError:
            pass
        return text
    return str(value)


def _ensure_span_id_column(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize Phoenix span query results to include a span_id column."""
    if df.empty:
        return df

    out = df.copy()
    if "span_id" in out.columns:
        return out

    if out.index.name in {"span_id", "context.span_id"}:
        out = out.reset_index()

    if "context.span_id" in out.columns and "span_id" not in out.columns:
        out["span_id"] = out["context.span_id"]

    return out


def fetch_generate_summary_spans(px_client: Any, *, project_name: str = PROJECT_NAME) -> pd.DataFrame:
    """Pull generate_summary tool spans from Phoenix using SpanQuery."""
    query = (
        SpanQuery()
        .select("span_id", "name", "span_kind", SpanAttributes.OUTPUT_VALUE)
        .rename(**{SpanAttributes.OUTPUT_VALUE: "summary"})
    )

    spans_df = pd.DataFrame()
    for project in (project_name, None):
        kwargs: dict[str, Any] = {"query": query, "limit": 5000}
        if project:
            kwargs["project_name"] = project
        candidate = px_client.spans.get_spans_dataframe(**kwargs)
        if not candidate.empty:
            spans_df = candidate
            break

    if spans_df.empty:
        return spans_df

    if "name" in spans_df.columns:
        spans_df = spans_df[spans_df["name"] == "generate_summary"]

    spans_df = spans_df.copy()
    spans_df["summary"] = spans_df["summary"].apply(_parse_summary_output)
    spans_df = spans_df.dropna(subset=["summary"]).copy()
    return _ensure_span_id_column(spans_df)


def log_groundedness_annotations_to_phoenix(
    px_client: Any,
    runs_df: pd.DataFrame,
    *,
    project_name: str = PROJECT_NAME,
    judge_model: str = "gpt-4o-mini",
) -> pd.DataFrame:
    """
    Query summary spans, join run metrics, and log LLM-as-judge scores to Phoenix UI.

    Uses scores already captured in runs_df when available; otherwise runs the judge.
    """
    spans_df = fetch_generate_summary_spans(px_client, project_name=project_name)
    if spans_df.empty:
        print(
            "No generate_summary spans found. Run the experiment cell in the same kernel session "
            "as launch_app()/setup_tracing(), then re-run this cell. CSV run logs still work for section 5."
        )
        return spans_df

    runs_for_join = runs_df[["summary", "metrics", "llm_judge_score", "judge_explanation"]].copy()
    runs_for_join = runs_for_join.drop_duplicates(subset=["summary"])

    merged = spans_df.merge(runs_for_join, on="summary", how="left")
    merged = _ensure_span_id_column(merged)

    if "span_id" not in merged.columns:
        print("Spans found but span_id column missing; skipping Phoenix annotations.")
        return pd.DataFrame()

    for idx, row in merged.iterrows():
        if pd.notna(row.get("llm_judge_score")):
            continue

        metrics_raw = row.get("metrics", {})
        if isinstance(metrics_raw, str):
            metrics = json.loads(metrics_raw) if metrics_raw else {}
        else:
            metrics = metrics_raw or {}

        score, explanation = score_summary_groundedness(
            summary=row["summary"],
            metrics=metrics,
            judge_model=judge_model,
        )
        merged.at[idx, "llm_judge_score"] = score
        merged.at[idx, "judge_explanation"] = explanation

    annotations_df = pd.DataFrame(
        {
            "span_id": merged["span_id"],
            "score": merged["llm_judge_score"].astype(float),
            "label": merged["llm_judge_score"].apply(
                lambda value: "grounded" if float(value) >= 0.5 else "not_grounded"
            ),
            "explanation": merged["judge_explanation"].fillna(""),
        }
    )

    px_client.spans.log_span_annotations_dataframe(
        dataframe=annotations_df,
        annotation_name="summary_groundedness",
        annotator_kind="LLM",
        sync=True,
    )
    return merged


class EvaluationLogger:
    """
    Log agent run metadata and LLM-as-judge scores.

    Phoenix handles trace logging automatically; this logger stores experiment-level
    records for causal analysis across prompt versions, temperatures, and models.
    """

    def __init__(self, log_path: str | Path = DEFAULT_LOG_PATH) -> None:
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log_run(
        self,
        *,
        run_id: str | None = None,
        prompt_version: str,
        temperature: float,
        input_tokens: int,
        tool_calls_made: list[str],
        task_success: bool,
        llm_judge_score: float,
        cost_usd: float,
        model: str = "gpt-4o-mini",
        summary: str = "",
        metrics: dict[str, Any] | None = None,
        judge_explanation: str = "",
        timestamp: datetime | None = None,
    ) -> dict[str, Any]:
        """Append one run record and return the logged row."""
        record = {
            "run_id": run_id or str(uuid.uuid4()),
            "prompt_version": prompt_version,
            "temperature": temperature,
            "model": model,
            "input_tokens": input_tokens,
            "tool_calls_made": json.dumps(tool_calls_made),
            "task_success": task_success,
            "llm_judge_score": llm_judge_score,
            "cost_usd": cost_usd,
            "timestamp": (timestamp or datetime.now(timezone.utc)).isoformat(),
            "summary": summary,
            "metrics": json.dumps(metrics or {}),
            "judge_explanation": judge_explanation,
        }

        row_df = pd.DataFrame([record])
        if self.log_path.exists():
            row_df.to_csv(self.log_path, mode="a", header=False, index=False)
        else:
            row_df.to_csv(self.log_path, index=False)

        return record

    def log_agent_result(
        self,
        agent_result: dict[str, Any],
        *,
        judge_model: str = "gpt-4o-mini",
        run_id: str | None = None,
    ) -> dict[str, Any]:
        """Score an agent result with LLM-as-judge and persist the run."""
        score, explanation = score_summary_groundedness(
            summary=agent_result.get("summary", ""),
            metrics=agent_result.get("metrics", {}),
            judge_model=judge_model,
        )

        return self.log_run(
            run_id=run_id,
            prompt_version=agent_result.get("prompt_version", "unknown"),
            temperature=float(agent_result.get("temperature", 0.0)),
            model=agent_result.get("model", "gpt-4o-mini"),
            input_tokens=int(agent_result.get("input_tokens", 0)),
            tool_calls_made=agent_result.get("tool_calls_made", []),
            task_success=bool(agent_result.get("task_success", False)),
            llm_judge_score=score,
            cost_usd=float(agent_result.get("cost_usd", 0.0)),
            summary=agent_result.get("summary", ""),
            metrics=agent_result.get("metrics", {}),
            judge_explanation=explanation,
        )

    def load_runs(self) -> pd.DataFrame:
        """Load all logged runs as a DataFrame."""
        if not self.log_path.exists():
            return pd.DataFrame()
        return pd.read_csv(self.log_path)

    def summarize_by_prompt_version(self) -> pd.DataFrame:
        """Aggregate task success and judge scores by prompt version."""
        runs = self.load_runs()
        if runs.empty:
            return runs

        return (
            runs.groupby("prompt_version", as_index=False)
            .agg(
                runs=("run_id", "count"),
                task_success_rate=("task_success", "mean"),
                avg_judge_score=("llm_judge_score", "mean"),
                avg_cost_usd=("cost_usd", "mean"),
            )
            .sort_values("prompt_version")
        )
