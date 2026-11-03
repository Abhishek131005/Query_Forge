"""Sequential, reproducible QueryForge benchmark runner.

Run from the repository root:
    python -m evaluation.runner --limit 3
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from backend.model_service import (
    DO_SAMPLE,
    MAX_NEW_TOKENS,
    PROMPT_VERSION,
    SQLModelService,
)
from frontend.sql_executor import execute_sql, validate_sql

from evaluation.metrics import aggregate_metrics, paired_outcomes
from evaluation.result_compare import compare_results


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_PATH = ROOT / "evaluation" / "benchmark.json"
RESULTS_DIR = ROOT / "evaluation" / "results"


def load_benchmark(path=BENCHMARK_PATH):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    records = payload["records"]
    if not 30 <= len(records) <= 50:
        raise ValueError("Benchmark must contain 30-50 records.")
    ids = [record["id"] for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Benchmark question IDs must be unique.")
    return payload


def build_fixture(payload, directory):
    db_path = Path(directory) / "benchmark.sqlite3"
    with sqlite3.connect(db_path) as connection:
        for fixture in payload["fixtures"]:
            csv_path = ROOT / "evaluation" / fixture
            with csv_path.open(newline="", encoding="utf-8") as handle:
                reader = csv.reader(handle)
                headers = next(reader)
                rows = list(reader)
            table = csv_path.stem
            types = ["INTEGER" if value.isdigit() else "TEXT" for value in rows[0]]
            columns = ", ".join(
                f'"{name}" {column_type}'
                for name, column_type in zip(headers, types)
            )
            connection.execute(f'CREATE TABLE "{table}" ({columns})')
            placeholders = ", ".join("?" for _ in headers)
            connection.executemany(
                f'INSERT INTO "{table}" VALUES ({placeholders})',
                [[int(value) if value.isdigit() else value for value in row] for row in rows],
            )
    return db_path


def verify_references(payload, db_path):
    for record in payload["records"]:
        frame, _, truncated = execute_sql(record["reference_sql"], db_path)
        if truncated:
            raise ValueError(f"Reference query is truncated: {record['id']}")
        record["_expected_frame"] = frame


def evaluate_model(service, model, records, schema, db_path, schema_id):
    evaluated = []
    for record in records:
        result = {
            "question_id": record["id"],
            "question": record["question"],
            "difficulty": record["difficulty"],
            "tags": ",".join(record["tags"]),
            "model": model,
            "reference_sql": record["reference_sql"],
            "schema_id": schema_id,
            "order_sensitive": record["order_sensitive"],
            "status": "model_error",
            "error": None,
            "raw_output": None,
            "extracted_sql": None,
            "inference_time_seconds": None,
            "generated_tokens": None,
            "parse_valid": False,
            "safety_accepted": False,
            "execution_success": False,
        }
        try:
            generation = service.generate_sql(record["question"], schema, model)
            result.update({
                "raw_output": generation["raw_output"],
                "extracted_sql": generation["extracted_sql"],
                "inference_time_seconds": generation["inference_time_seconds"],
                "generated_tokens": generation["generated_tokens"],
            })
            try:
                validate_sql(generation["extracted_sql"])
            except ValueError as exc:
                message = str(exc)
                unsafe = (
                    "Only SELECT" in message
                    or "prohibited" in message
                    or "commands" in message
                )
                result.update({
                    "status": "unsafe_sql" if unsafe else "invalid_sql",
                    "parse_valid": unsafe,
                    "error": message,
                })
                evaluated.append(result)
                continue
            result["parse_valid"] = True
            result["safety_accepted"] = True
            try:
                actual, execution_time, truncated = execute_sql(
                    generation["extracted_sql"], db_path
                )
            except ValueError as exc:
                result.update({"status": "execution_error", "error": str(exc)})
                evaluated.append(result)
                continue
            result["execution_time_seconds"] = execution_time
            result["execution_success"] = True
            if truncated:
                result.update({"status": "inconclusive", "error": "result_truncated"})
            elif compare_results(record["_expected_frame"], actual, record["order_sensitive"]):
                result["status"] = "correct"
            else:
                result["status"] = "incorrect"
        except Exception as exc:
            result["error"] = str(exc)
        evaluated.append(result)
    return evaluated


def _metadata(payload, run_id, records, service):
    return {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "benchmark_version": payload["benchmark_version"],
        "fixture_version": payload["fixture_version"],
        "benchmark_sha256": hashlib.sha256(
            BENCHMARK_PATH.read_bytes()
        ).hexdigest(),
        "git_commit": _git_commit(),
        "python_version": sys.version,
        "model_class": type(service).__name__,
        "base_model_id": getattr(service, "base_model_id", None),
        "adapter_path": str(getattr(service, "ADAPTER_PATH", ROOT / "models" / "tinyllama-sql-lora-adapter")),
        "device": getattr(service, "device", None),
        "prompt_version": PROMPT_VERSION,
        "generation_config": {
            "max_new_tokens": MAX_NEW_TOKENS,
            "do_sample": DO_SAMPLE,
        },
        "sample_count": len(records),
    }


def run(limit=None, output_dir=RESULTS_DIR):
    payload = load_benchmark()
    records = payload["records"][:limit] if limit else payload["records"]
    run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as directory:
        db_path = build_fixture(payload, directory)
        verify_references(payload, db_path)
        schema = (ROOT / "evaluation" / payload["schema_file"]).read_text(encoding="utf-8")
        service = SQLModelService()
        base_records = evaluate_model(
            service, "base", records, schema, db_path, payload["schema_id"]
        )
        finetuned_records = evaluate_model(
            service, "finetuned", records, schema, db_path, payload["schema_id"]
        )
    artifact = {
        "metadata": _metadata(payload, run_id, records, service),
        "models": {
            "base": {"metrics": aggregate_metrics(base_records), "records": base_records},
            "finetuned": {
                "metrics": aggregate_metrics(finetuned_records),
                "records": finetuned_records,
            },
            "paired_outcomes": paired_outcomes(base_records, finetuned_records),
        },
    }
    json_path = output_dir / f"{run_id}.json"
    csv_path = output_dir / f"{run_id}.csv"
    _atomic_write(json_path, json.dumps(artifact, indent=2, default=str))
    rows = base_records + finetuned_records
    _atomic_write_csv(csv_path, rows)
    return json_path, csv_path


def _atomic_write(path, content):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _atomic_write_csv(path, rows):
    fields = sorted({key for row in rows for key in row})
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, help="Run a smoke/development subset.")
    args = parser.parse_args()
    json_path, csv_path = run(limit=args.limit)
    print(f"Saved JSON: {json_path}")
    print(f"Saved CSV: {csv_path}")
