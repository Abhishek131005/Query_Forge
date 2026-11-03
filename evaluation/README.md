# QueryForge benchmark

This package contains the versioned `employees_departments_v1` benchmark and
its local fixture. The fixture has two related tables and 30 curated questions
covering filters, projections, aggregations, grouping, joins, subqueries,
sorting, limits, dates, and common predicates.

## Commands

From the repository root with the project virtual environment active:

```powershell
$env:PYTHONPATH = "."
python -m unittest evaluation.test_evaluation frontend.test_sql_executor backend.test_backend_contract -v
python -m evaluation.runner --limit 3
python -m evaluation.runner
```

The first runner command is the required 3-question smoke run. Both commands
load TinyLlama and run base and fine-tuned inference sequentially, so CPU
execution may take substantial time. The runner never repairs generated SQL
before scoring.

Each run writes a unique JSON artifact and CSV artifact under
`evaluation/results/`. These files are ignored by Git and contain run
metadata, raw completions, extracted SQL, errors, measured timings, token
counts, statuses, and aggregate metrics. They do not contain fixture rows or
uploaded user data.

Metrics use every selected benchmark question as the denominator. Latency and
token averages use only completed generations and report their sample count.
Execution accuracy requires the generated result set to match the verified
reference result, including columns, duplicates, NULLs, numeric tolerance, and
the record's ordering policy.

## Limitations

The benchmark is a local regression fixture, not evidence of general
production accuracy. It does not measure throughput under concurrency, model
quality outside this schema, or security isolation. Results are not created
until the runner completes actual inference; no sample metrics are checked in.
