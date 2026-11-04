# QueryForge AI

QueryForge is a local Text-to-SQL application:

```text
Streamlit -> FastAPI -> TinyLlama + LoRA -> SQLGlot -> read-only SQLite -> Plotly
```

The frontend supports three pages: AI SQL Studio, Database Explorer, and Model
Comparison. Demo mode uses a clearly labeled built-in employees dataset.
Uploaded CSVs require a running inference backend for generation; manually
entered SQL remains available.

## Windows setup

Use Python 3.10 in a virtual environment. From PowerShell:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r backend\requirements.txt
python -m pip install -r frontend\requirements.txt
```

The backend requirements select CPU-only PyTorch. Use an approved CUDA
installation only when GPU inference is intentionally configured. Model
weights are not downloaded by the test suite; the local adapter is expected at
`models\tinyllama-sql-lora-adapter`.

## Run

Terminal 1:

```powershell
.\.venv\Scripts\Activate.ps1
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Terminal 2:

```powershell
.\.venv\Scripts\Activate.ps1
$env:INFERENCE_API_URL = "http://127.0.0.1:8000"
python -m streamlit run frontend\app.py
```

Open the displayed Streamlit URL. The sidebar health indicator distinguishes a
loaded backend from a configured-but-unavailable endpoint.

## Deploy

The recommended hosted setup uses Streamlit Community Cloud for the frontend
and Render for the FastAPI inference backend.

1. Push this repository to GitHub.
2. In Render, create the web service from this repository. The included
   `render.yaml` installs the backend dependencies, starts FastAPI on Render's
   assigned port, and configures `/health` as the health check.
3. Wait for the Render service to report healthy, then copy its public HTTPS
   URL.
4. In Streamlit Community Cloud, deploy `frontend/app.py` and set the
   `INFERENCE_API_URL` secret to the Render URL, without a trailing slash.
5. Open the Streamlit app and confirm the sidebar reports that the backend is
   healthy and the model is loaded.

The model adapter is loaded by the backend at startup, so the first deploy and
first request can take several minutes. Do not expose the backend without
authentication for sensitive or multi-user workloads; this project is designed
as a single-user demonstration app.

## Tests and benchmark

```powershell
$env:PYTHONPATH = "."
python -m unittest evaluation.test_evaluation frontend.test_sql_executor `
  backend.test_backend_contract frontend.test_dashboard -v
python -m evaluation.runner --limit 3
```

The benchmark smoke run and full run perform actual sequential base and
fine-tuned inference. Results are saved as unique JSON/CSV files under
`evaluation\results\` and are never fabricated or overwritten silently.

## Limits and honest interpretation

CSV uploads are limited to 10 MB, 100,000 rows, and 100 columns. SQL is
restricted to read-only analytical statements, has a five-second progress
timeout, and displays at most 500 rows. These controls are not a complete
multi-tenant security sandbox.

Execution success is not correctness; benchmark accuracy requires matching the
trusted reference result set. CPU latency depends on local hardware and should
not be generalized from a small run.
