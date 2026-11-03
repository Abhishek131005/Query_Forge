"""Metric aggregation with explicit all-question denominators."""
from __future__ import annotations

from collections import Counter, defaultdict


def aggregate_metrics(records):
    total = len(records)
    categories = Counter(record.get("status", "model_error") for record in records)
    parse_valid = sum(bool(record.get("parse_valid")) for record in records)
    safety_accepted = sum(bool(record.get("safety_accepted")) for record in records)
    execution_success = sum(bool(record.get("execution_success")) for record in records)
    metrics = {
        "total_questions": total,
        "parse_validity_rate": _rate(parse_valid, total),
        "safety_acceptance_rate": _rate(safety_accepted, total),
        "execution_success_rate": _rate(execution_success, total),
        "execution_accuracy": _rate(categories["correct"], total),
        "status_counts": dict(sorted(categories.items())),
    }

    latencies = [
        record["inference_time_seconds"]
        for record in records
        if isinstance(record.get("inference_time_seconds"), (int, float))
    ]
    tokens = [
        record["generated_tokens"]
        for record in records
        if isinstance(record.get("generated_tokens"), int)
    ]
    metrics["latency_sample_count"] = len(latencies)
    metrics["mean_inference_time_seconds"] = (
        sum(latencies) / len(latencies) if latencies else None
    )
    metrics["mean_generated_tokens"] = (
        sum(tokens) / len(tokens) if tokens else None
    )
    return metrics


def paired_outcomes(base_records, finetuned_records):
    base = {record["question_id"]: record.get("status") == "correct" for record in base_records}
    finetuned = {
        record["question_id"]: record.get("status") == "correct"
        for record in finetuned_records
    }
    counts = Counter()
    for question_id in sorted(set(base) | set(finetuned)):
        if base.get(question_id) and finetuned.get(question_id):
            counts["both_correct"] += 1
        elif base.get(question_id):
            counts["base_only_correct"] += 1
        elif finetuned.get(question_id):
            counts["finetuned_only_correct"] += 1
        else:
            counts["neither_correct"] += 1
    return dict(sorted(counts.items()))


def _rate(numerator, denominator):
    return numerator / denominator if denominator else None
