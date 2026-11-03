# QueryForge AI: Streamlit starter

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
```

Without an inference URL, this runs in clearly labeled **demo mode** with a tiny in-memory employees database and hard-coded sample SQL responses. It does **not** load your fine-tuned TinyLlama adapter.

## Run the local backend

From the repository root, install the backend dependencies and start the API:

```powershell
python -m pip install -r .\backend\requirements.txt
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Verify it at `http://127.0.0.1:8000/health`. The API documentation is available at
`http://127.0.0.1:8000/docs`. A `404` from the root URL is not expected after
the backend is updated; the root URL now returns API status information.

In the Streamlit sidebar, set **Backend URL** to:

```text
http://127.0.0.1:8000
```

## Connect your real LoRA model
Host TinyLlama plus your saved `tinyllama-sql-lora-adapter` on a suitable inference server. Expose:

`POST /generate-sql`

Request:
```json
{"question":"List Engineering employees", "schema":"CREATE TABLE employees (...) ;", "model":"finetuned"}
```
Response:
```json
{"sql":"SELECT * FROM employees WHERE department = 'Engineering';"}
```
Set `INFERENCE_API_URL` or enter the URL in the sidebar.

## Safety and limits
The SQL executor accepts one read-only `SELECT`/`UNION` statement, including
read-only CTEs and subqueries. It uses SQLGlot validation, a SQLite read-only
URI, `PRAGMA query_only`, an authorizer, a five-second progress timeout, and a
500-row display cap. Truncated results are marked in the UI and history.

These controls are **not a production or multi-tenant security sandbox**.
SQLite read-only mode and query timeouts do not isolate a malicious process,
prevent all denial-of-service patterns, or replace operating-system
permissions. For public traffic, run queries in a separate constrained
process/container with explicit resource limits and access controls.

History is session-scoped and is not durable across browser sessions. CSV and
JSON exports contain query metadata and SQL provenance, not result rows or
uploaded dataset contents by default.
