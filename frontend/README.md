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
The included SQL executor is only for a fixed trusted demo SQLite database. It accepts a single SELECT, blocks common mutation keywords and caps displayed results. This regex check is **not a production SQL security boundary**. For real user-supplied databases, use an isolated read-only database role, a proper SQL parser/allowlist, query timeouts, row limits, and strict access controls. Uploaded CSVs are preview-only in this starter.
