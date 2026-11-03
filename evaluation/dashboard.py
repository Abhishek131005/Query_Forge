"""Helpers for loading and summarizing saved benchmark runs."""
from __future__ import annotations

import json
from pathlib import Path


def list_run_files(results_dir):
    return sorted(
        Path(results_dir).glob("*.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )


def load_run(path):
    with Path(path).open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload.get("metadata"), dict) or not isinstance(
        payload.get("models"), dict
    ):
        raise ValueError("Evaluation artifact has an invalid structure.")
    return payload


def model_rows(run, model):
    return run.get("models", {}).get(model, {}).get("records", [])


def filter_rows(rows, difficulty="All", tag="All", status="All"):
    filtered = rows
    if difficulty != "All":
        filtered = [row for row in filtered if row.get("difficulty") == difficulty]
    if tag != "All":
        filtered = [
            row for row in filtered
            if tag in str(row.get("tags", "")).split(",")
        ]
    if status != "All":
        filtered = [row for row in filtered if row.get("status") == status]
    return filtered


def combined_rows(run, difficulty="All", tag="All", status="All"):
    rows = []
    for model in ("base", "finetuned"):
        rows.extend(
            filter_rows(model_rows(run, model), difficulty, tag, status)
        )
    return rows
