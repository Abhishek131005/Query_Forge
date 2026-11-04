from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sql_executor import execute_sql
from database import database_path
from datetime import datetime, timezone
import json
import os
import re
import sqlite3
import requests
import pandas as pd
import plotly.express as px
from database import (
    import_csv,
    tables,
    schema_sql,
    table_info,
    delete_table,
    clean_identifier,
    connect,
    cleanup_abandoned_databases,
)
from evaluation.result_compare import compare_results
import streamlit as st

st.set_page_config(page_title='QueryForge AI', page_icon='⚡', layout='wide')
st.markdown('''<style>
.stApp {background:#0b1020;color:#e9ecf7} [data-testid="stSidebar"]{background:#11182c}
.block-container{padding:2rem 3rem 4rem;max-width:1500px} h1,h2,h3{letter-spacing:-.035em}
.hero{padding:32px;border:1px solid #293451;border-radius:20px;background:linear-gradient(120deg,#192448,#17142d);margin-bottom:1.5rem}
.small{color:#aeb9d2}
.stTextArea textarea,.stTextInput input {
    color:#172033 !important;
    background-color:#f8fafc !important;
    caret-color:#172033 !important;
}
.stTextArea label,.stTextInput label {
    color:#e9ecf7 !important;
}
.stButton button{border-radius:10px;min-height:2.6rem}
[data-testid="stMetric"]{background:#121d38;border:1px solid #293451;border-radius:12px;padding:12px}
[data-testid="stDataFrame"]{border:1px solid #293451;border-radius:12px}
.result-card{background:#111b34;border:1px solid #293451;border-radius:14px;padding:1rem}
</style>''', unsafe_allow_html=True)

@st.cache_resource
def demo_db():
    conn=sqlite3.connect(':memory:',check_same_thread=False)
    conn.execute('CREATE TABLE employees (id INTEGER PRIMARY KEY, name TEXT, department TEXT, salary INTEGER)')
    conn.executemany('INSERT INTO employees VALUES (?,?,?,?)',[(1,'Asha','Engineering',120000),(2,'Rahul','Sales',75000),(3,'Mira','Engineering',145000),(4,'Neha','Finance',92000),(5,'Kabir','Engineering',85000)])
    conn.commit()
    return conn

if 'history' not in st.session_state: st.session_state.history=[]
for key, default in {
    'question': 'List Engineering employees earning more than 100000.',
    'question_widget': None,
    'comparison_question_widget': None,
    'comparison_outputs': {},
    'comparison_reference_sql': '',
    'comparison_rows': None,
    'query_result_data': None,
    'schema': '',
    'sql_editor': '',
    'query_result': None,
    'execution_time': None,
    'result_truncated': False,
    'executed_sql': None,
    'generated_sql': None,
    'generation_metadata': {},
    'pending_sql_load': None,
    'results_stale': False,
    'is_demo': False,
}.items():
    st.session_state.setdefault(key, default)


def clear_query_state(clear_editor=False):
    st.session_state.query_result = None
    st.session_state.query_result_data = None
    st.session_state.execution_time = None
    st.session_state.result_truncated = False
    st.session_state.executed_sql = None
    st.session_state.results_stale = False
    if clear_editor:
        st.session_state.sql_editor = ''
        st.session_state.generated_sql = None
        st.session_state.generation_metadata = {}


def record_history(event, **details):
    st.session_state.history.append({
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'event': event,
        **details,
    })
    st.session_state.history = st.session_state.history[-100:]


def format_timestamp(value):
    try:
        timestamp = datetime.fromisoformat(value)
        return timestamp.astimezone().strftime('%Y-%m-%d %H:%M:%S %Z')
    except (TypeError, ValueError):
        return str(value)


def history_title(item):
    event = item.get('event', 'activity').replace('_', ' ').title()
    status = item.get('execution_status')
    if status:
        status = status.replace('_', ' ').title()
        event = f'{event} - {status}'
    question = str(item.get('question', '')).strip()
    if question:
        question = question if len(question) <= 70 else question[:67] + '...'
        return f'{event} | {question}'
    return event


with st.sidebar:
    st.title('⚡ QueryForge AI')
    page=st.radio('Workspace',['AI SQL Studio','Database Explorer','Model Comparison'])
    previous_page = st.session_state.get('active_page')
    if page != previous_page:
        if page == 'AI SQL Studio':
            st.session_state.question_widget = st.session_state.question
        elif page == 'Model Comparison':
            st.session_state.comparison_question_widget = st.session_state.question
    st.session_state.active_page = page
    st.divider()
    st.caption('Inference endpoint')
    endpoint=st.text_input('Backend URL',value=os.getenv('INFERENCE_API_URL','http://127.0.0.1:8000'),help='Use a blank value for deterministic demo mode.')
    if endpoint:
        try:
            health = requests.get(endpoint.rstrip('/') + '/health', timeout=2).json()
            if health.get('model_loaded'):
                st.caption('Backend healthy: model loaded.')
            else:
                st.warning('Backend reachable, but model is not loaded.')
        except requests.RequestException:
            st.warning('Backend URL configured, but health check failed.')
    else:
        st.caption('Demo mode: deterministic sample queries are used.')

# Demo database is used only until a user imports a CSV.
workspace_path = database_path()
cleanup_abandoned_databases(workspace_path)
active_tables = tables()
conn = None if active_tables else demo_db()
active_schema = schema_sql() if active_tables else 'CREATE TABLE employees (id INTEGER PRIMARY KEY, name TEXT, department TEXT, salary INTEGER);'

previous_schema = st.session_state.get('active_schema_snapshot')
database_changed = (
    previous_schema is not None and previous_schema != active_schema
)
if database_changed:
    clear_query_state(clear_editor=True)
st.session_state.active_schema_snapshot = active_schema

if database_changed:
    st.session_state.schema = active_schema
elif not st.session_state.schema or (
    st.session_state.schema == st.session_state.get('schema_source')
):
    st.session_state.schema = active_schema
st.session_state.schema_source = active_schema
schema = st.session_state.schema
if st.session_state.pending_sql_load is not None:
    st.session_state.sql_editor = st.session_state.pending_sql_load
    st.session_state.pending_sql_load = None
if (
    st.session_state.query_result is None
    and st.session_state.query_result_data is not None
):
    saved_result = st.session_state.query_result_data
    st.session_state.query_result = pd.DataFrame(
        saved_result['rows'],
        columns=saved_result['columns'],
    )

with st.sidebar:
    st.divider()
    st.caption(f'Active database: {len(active_tables)} uploaded table(s)' if active_tables else 'Active database: demo employees')

def generate_sql(question,schema,model='finetuned'):
    if not endpoint:
        if active_tables:
            raise ValueError(
                'Demo SQL is available only for the built-in demo schema. '
                'Connect the inference backend or enter SQL manually.'
            )
        q=question.lower()
        if ('department' in q or 'each' in q) and ('count' in q or 'how many' in q):
            sql = 'SELECT department, COUNT(*) AS employee_count FROM employees GROUP BY department;'
            return sql,True, {
                'model': 'demo',
                'raw_output': sql,
                'extracted_sql': sql,
                'postprocessed': False,
            }
        if 'average' in q or 'avg' in q:
            sql = 'SELECT department, AVG(salary) AS average_salary FROM employees GROUP BY department;'
            return sql,True, {
                'model': 'demo',
                'raw_output': sql,
                'extracted_sql': sql,
                'postprocessed': False,
            }
        if 'engineering' in q and ('salary' in q or 'earning' in q):
            threshold = re.search(r'(?:more than|over|above|greater than)\s+(\d+)', q)
            amount = threshold.group(1) if threshold else '100000'
            sql = f"SELECT name, salary FROM employees WHERE department = 'Engineering' AND salary > {amount};"
            return sql,True, {
                'model': 'demo',
                'raw_output': sql,
                'extracted_sql': sql,
                'postprocessed': False,
            }
        if 'engineering' in q:
            sql = "SELECT * FROM employees WHERE department = 'Engineering';"
        else:
            sql = 'SELECT * FROM employees LIMIT 10;'
        return sql,True, {
            'model': 'demo',
            'raw_output': sql,
            'extracted_sql': sql,
            'postprocessed': False,
        }
    response=requests.post(endpoint.rstrip('/')+'/generate-sql',json={'question':question,'schema':schema,'model':model},timeout=90)
    response.raise_for_status()
    result = response.json()
    sql = result.get('sql')
    if not isinstance(sql, str) or not sql.strip():
        raise ValueError('The backend returned no SQL query.')
    return sql.strip(),False,result

def safe_query(sql):
    if active_tables:
        return execute_sql(sql, database_path())

    # Demo database remains read-only for execution.
    # Use an isolated temporary SQLite database instead
    # of the shared in-memory demo connection.
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "demo.sqlite3"

        demo_conn = sqlite3.connect(path)
        try:
            demo_conn.execute("""
                CREATE TABLE employees (
                    id INTEGER PRIMARY KEY,
                    name TEXT,
                    department TEXT,
                    salary INTEGER
                )
            """)

            demo_conn.executemany(
                "INSERT INTO employees VALUES (?, ?, ?, ?)",
                [
                    (1, "Asha", "Engineering", 120000),
                    (2, "Rahul", "Sales", 75000),
                    (3, "Mira", "Engineering", 145000),
                    (4, "Neha", "Finance", 92000),
                    (5, "Kabir", "Engineering", 85000)
                ]
            )
            demo_conn.commit()
        finally:
            demo_conn.close()

        return execute_sql(sql, path)

if page=='AI SQL Studio':
    st.markdown('<div class="hero"><h1>Ask your data anything.</h1><p class="small">Turn natural language into safe, read-only SQL and explore the result.</p></div>',unsafe_allow_html=True)
    st.write('')
    a,b=st.columns([1,1])
    with a:
        st.subheader('Your question')
        if st.session_state.question_widget is None:
            st.session_state.question_widget = st.session_state.question
        question=st.text_area('Ask in plain English',height=110,key='question_widget')
        st.session_state.question = question
        user_schema=st.text_area('Database schema (auto-generated from your CSV tables)',height=145,key='schema')
        if user_schema != active_schema:
            st.warning(
                'The schema text differs from the active SQLite database. '
                'Generation will use the schema text shown here.'
            )
        run = st.button(
            "⚡ Generate SQL",
            type="primary",
            use_container_width=True
        )
    with b:
        st.subheader('Generated SQL')
        if run:
            clear_query_state(clear_editor=True)
            try:
                with st.spinner('Generating SQL...'):
                    sql,demo,metadata=generate_sql(question,user_schema)
                st.session_state.sql_editor=sql
                st.session_state.generated_sql=sql
                st.session_state.generation_metadata=metadata
                st.session_state.is_demo=demo
                record_history(
                    'generated',
                    question=question,
                    schema=user_schema,
                    model=metadata.get('model', 'demo' if demo else 'unknown'),
                    database_id=workspace_path.name,
                    raw_output=metadata.get('raw_output', sql),
                    cleaned_output=metadata.get('cleaned_output', sql),
                    extracted_sql=metadata.get('extracted_sql', sql),
                    edited_sql=None,
                    execution_status='not_executed',
                    row_count=None,
                    inference_latency=metadata.get('inference_time_seconds'),
                    execution_latency=None,
                    error=None,
                    result_truncated=False,
                    demo=demo,
                )
            except Exception as e:
                st.session_state.is_demo = False
                st.error(f'Inference failed: {e}')
                record_history(
                    'failed',
                    operation='generation',
                    question=question,
                    schema=user_schema,
                    model='demo' if not endpoint else 'unknown',
                    database_id=workspace_path.name,
                    raw_output=None,
                    cleaned_output=None,
                    extracted_sql=None,
                    edited_sql=None,
                    execution_status='failed',
                    row_count=None,
                    inference_latency=None,
                    execution_latency=None,
                    error=str(e),
                    result_truncated=False,
                )
        sql=st.text_area('SQL editor',height=160,key='sql_editor')
        if (
            st.session_state.executed_sql is not None
            and sql != st.session_state.executed_sql
            and st.session_state.query_result is not None
        ):
            st.session_state.query_result = None
            st.session_state.query_result_data = None
            st.session_state.execution_time = None
            st.session_state.result_truncated = False
            st.session_state.results_stale = True
        if st.session_state.results_stale:
            st.warning(
                'The SQL editor was changed after the last execution. '
                'Displayed results were cleared; execute this SQL to refresh them.'
            )
        if st.session_state.get('is_demo',not endpoint):st.info('Demo response. Connect a model backend for real inference.')
        execute=st.button('▶ Execute SQL',disabled=not bool(sql.strip()),use_container_width=True)
    if execute:
        try:
            with st.spinner('Executing read-only SQL...'):
                df, execution_time, truncated = safe_query(sql)

            st.session_state.query_result = df
            st.session_state.query_result_data = {
                'columns': df.columns.tolist(),
                'rows': df.to_dict(orient='records'),
            }
            st.session_state.execution_time = execution_time
            st.session_state.result_truncated = truncated
            st.session_state.executed_sql = sql
            st.session_state.results_stale = False
            record_history(
                'executed',
                question=question,
                schema=user_schema,
                model=st.session_state.generation_metadata.get('model'),
                database_id=workspace_path.name,
                raw_output=st.session_state.generation_metadata.get('raw_output'),
                cleaned_output=st.session_state.generation_metadata.get('cleaned_output'),
                extracted_sql=st.session_state.generation_metadata.get('extracted_sql'),
                edited_sql=(
                    sql
                    if sql != st.session_state.generated_sql
                    else None
                ),
                execution_status='succeeded',
                row_count=len(df),
                inference_latency=st.session_state.generation_metadata.get(
                    'inference_time_seconds'
                ),
                execution_latency=execution_time,
                error=None,
                result_truncated=truncated,
            )

            st.success(f"Query executed successfully — {len(df):,} row(s) returned.")

        except Exception as e:
            clear_query_state()
            record_history(
                'failed',
                operation='execution',
                question=question,
                schema=user_schema,
                model=st.session_state.generation_metadata.get('model'),
                database_id=workspace_path.name,
                raw_output=st.session_state.generation_metadata.get('raw_output'),
                cleaned_output=st.session_state.generation_metadata.get('cleaned_output'),
                extracted_sql=st.session_state.generation_metadata.get('extracted_sql'),
                edited_sql=sql if sql != st.session_state.generated_sql else None,
                execution_status='failed',
                row_count=None,
                inference_latency=st.session_state.generation_metadata.get(
                    'inference_time_seconds'
                ),
                execution_latency=None,
                error=str(e),
                result_truncated=False,
            )
            st.error(f"Query execution failed: {e}")

    if (
        st.session_state.query_result is not None
        and not st.session_state.results_stale
    ):

        df = st.session_state.query_result

        st.divider()
        st.subheader("📊 Query Results")

        col1, col2, col3 = st.columns(3)

        col1.metric("Rows Returned", len(df))
        col2.metric("Columns", len(df.columns))
        col3.metric(
            "Execution Time",
            f"{st.session_state.execution_time:.3f} sec"
        )

        if st.session_state.get("result_truncated"):
            st.warning(
                "Results were limited to 500 rows. "
                "Add filters or aggregation to your query."
            )

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            label="⬇ Download Results as CSV",
            data=df.to_csv(index=False),
            file_name="queryforge_results.csv",
            mime="text/csv"
    )
        st.divider()
    # Keep the visualization section available before the first query runs.
    df = (
        st.session_state.query_result
        if st.session_state.query_result is not None
        and not st.session_state.results_stale
        else pd.DataFrame()
    )
    st.subheader("📈 Data Visualization")

    if df.empty:
        st.info("Run a query to unlock visualizations.")

    else:
        columns = df.columns.tolist()
        numeric_columns = df.select_dtypes(include="number").columns.tolist()
        categorical_columns = [column for column in columns if column not in numeric_columns]
        if not numeric_columns:
            st.info("This result contains no numeric column to plot. The table above is the complete result.")
        else:
            default_x = categorical_columns[0] if categorical_columns else columns[0]
            default_y = numeric_columns[0]
            controls = st.columns([1, 1, 1])
            with controls[0]:
                chart_type = st.selectbox("Chart", ["Bar Chart", "Line Chart", "Pie Chart", "Scatter Plot", "Histogram"])
            with controls[1]:
                x_axis = st.selectbox("Category / X-axis", columns, index=columns.index(default_x))
            with controls[2]:
                y_axis = st.selectbox("Value / Y-axis", numeric_columns, index=numeric_columns.index(default_y))

            try:
                if chart_type == "Histogram":
                    fig = px.histogram(df, x=y_axis, title=f"Distribution of {y_axis}", template="plotly_dark")
                elif chart_type == "Bar Chart":

                    fig = px.bar(
                        df,
                        x=x_axis,
                        y=y_axis,
                        title=f"{y_axis} by {x_axis}",
                        template="plotly_dark"
                    )

                elif chart_type == "Line Chart":

                    fig = px.line(
                        df,
                        x=x_axis,
                        y=y_axis,
                        markers=True,
                        template="plotly_dark"
                    )

                elif chart_type == "Pie Chart":

                    fig = px.pie(
                        df,
                        names=x_axis,
                        values=y_axis,
                        template="plotly_dark"
                    )

                else:
                    fig = px.scatter(
                        df,
                        x=x_axis,
                        y=y_axis,
                        template="plotly_dark"
                    )

                fig.update_layout(
                    paper_bgcolor="#0b1020",
                    plot_bgcolor="#0b1020",
                    font_color="#e9ecf7"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                    key=f"query-chart-{len(df)}-{chart_type}-{x_axis}-{y_axis}"
                )

            except Exception as e:
                st.error(f"Chart error: {e}")
    if st.session_state.history:
        st.subheader('History and provenance')
        history_frame = pd.DataFrame(st.session_state.history)
        st.download_button(
            'Download history (CSV)',
            history_frame.to_csv(index=False),
            file_name='queryforge_history.csv',
            mime='text/csv',
        )
        st.download_button(
            'Download history (JSON)',
            json.dumps(st.session_state.history, indent=2),
            file_name='queryforge_history.json',
            mime='application/json',
        )
        for index, item in enumerate(reversed(st.session_state.history[-5:])):
            with st.expander(
                f"{format_timestamp(item.get('timestamp'))} | "
                f"{item.get('model', 'unknown')} | {history_title(item)}",
                expanded=False,
            ):
                st.write(item.get('question', ''))
                if item.get('raw_output') is not None:
                    st.caption('Model output (prompt echoes removed)')
                    st.code(item.get('cleaned_output') or item['raw_output'])
                if item.get('extracted_sql'):
                    st.caption('Extracted SQL')
                    st.code(item['extracted_sql'], language='sql')
                if item.get('edited_sql'):
                    st.caption('User-edited SQL')
                    st.code(item['edited_sql'], language='sql')
                load_sql = item.get('edited_sql') or item.get('extracted_sql') or item.get('sql')
                if load_sql and st.button(
                    'Load SQL into editor',
                    key=f'load-history-{index}-{item["timestamp"]}',
                ):
                    st.session_state.pending_sql_load = load_sql
                    st.rerun()
                if item.get('error'):
                    st.error(item['error'])
elif page=='Database Explorer':
    st.title('Database Explorer')
    st.caption('Upload CSV files to create tables in your session-specific SQLite workspace.')
    uploads = st.file_uploader('Upload CSV datasets (up to 10 MB each)', type=['csv'], accept_multiple_files=True)
    if uploads:
        for upload in uploads:
            with st.form(f'import_{upload.name}'):
                proposed = clean_identifier(upload.name.rsplit('.', 1)[0])
                name = st.text_input(f'Table name for {upload.name}', value=proposed)
                submitted = st.form_submit_button(f'Import {upload.name}')
                if submitted:
                    try:
                        table, rows, columns = import_csv(upload, name)
                        clear_query_state(clear_editor=True)
                        st.success(f'Imported {table}: {rows:,} rows and {columns} columns')
                        st.rerun()
                    except Exception as exc:
                        st.error(str(exc))
    current_tables = tables()
    if current_tables:
        selected = st.selectbox('Select table', current_tables)
        columns, count, preview = table_info(selected)
        x, y = st.columns(2)
        x.metric('Rows', f'{count:,}')
        y.metric('Columns', len(columns))
        st.subheader('Columns and inferred SQLite types')
        st.dataframe(columns[['name', 'type', 'notnull', 'pk']], use_container_width=True, hide_index=True)
        st.subheader('Preview (first 50 rows)')
        st.dataframe(preview, use_container_width=True, hide_index=True)
        st.subheader('Generated schema for LLM')
        st.code(schema_sql(), language='sql')
        if st.button(f'Delete table {selected}', type='secondary'):
            delete_table(selected)
            clear_query_state(clear_editor=True)
            st.rerun()
        st.success(
            "Your database is ready! Open AI SQL Studio "
            "to generate and execute SQL queries."
        )
    else:
        st.info('No uploaded tables yet. The AI SQL Studio currently uses its built-in employees demo schema.')
        st.code(schema, language='sql')
        st.dataframe(pd.read_sql_query('SELECT * FROM employees', demo_db()), use_container_width=True, hide_index=True)
elif page=='Model Comparison':
    st.title('Base vs Fine-tuned')
    st.caption('Measured outputs use the same question and schema. No single-question result is benchmark accuracy.')
    if st.session_state.comparison_question_widget is None:
        st.session_state.comparison_question_widget = st.session_state.question
    question=st.text_input('Question',key='comparison_question_widget')
    st.session_state.question = question
    compare_clicked = st.button('Compare models',disabled=not bool(endpoint))
    if compare_clicked:
        outputs = {}
        for kind in ('base', 'finetuned'):
            try:
                with st.spinner(f'Generating {kind} output...'):
                    response = requests.post(
                        endpoint.rstrip('/') + '/generate-sql',
                        json={'question': question, 'schema': schema, 'model': kind},
                        timeout=90,
                    )
                    response.raise_for_status()
                    outputs[kind] = response.json()
            except requests.RequestException as exc:
                st.error(f'{kind.title()} request failed: {exc}')
            except Exception as exc:
                st.error(f'{kind.title()} response failed: {exc}')
        if outputs:
            st.session_state.comparison_outputs = outputs
            st.session_state.comparison_rows = None
            record_history(
                'compared',
                question=question,
                schema=schema,
                model='base and finetuned',
                database_id=workspace_path.name,
                raw_output={
                    kind: payload.get('raw_output')
                    for kind, payload in outputs.items()
                },
                extracted_sql={
                    kind: payload.get('extracted_sql', payload.get('sql'))
                    for kind, payload in outputs.items()
                },
                execution_status='not_executed',
                row_count=None,
                inference_latency={
                    kind: payload.get('inference_time_seconds')
                    for kind, payload in outputs.items()
                },
                execution_latency=None,
                error=None,
            )
    if st.session_state.comparison_outputs:
        cols=st.columns(2)
        for col,kind in zip(cols,['base','finetuned']):
            payload = st.session_state.comparison_outputs.get(kind)
            if not payload:
                continue
            with col:
                st.subheader(kind.title())
                st.caption(f"Latency: {payload.get('inference_time_seconds', 'Not measured')} sec")
                st.caption(f"Generated tokens: {payload.get('generated_tokens', 'Not measured')}")
                st.caption('Model output (prompt echoes removed)')
                st.code(
                    payload.get('cleaned_output')
                    or payload.get('extracted_sql')
                    or payload.get('sql', 'Not returned')
                )
                st.caption('Extracted SQL')
                st.code(payload.get('extracted_sql', payload.get('sql', 'Not returned')), language='sql')
        reference_sql = st.text_area(
            'Optional reference SQL for a single-question result check',
            key='comparison_reference_sql',
            placeholder='Provide a trusted reference query to compare result sets.',
        )
        if st.button('Evaluate both', disabled=not bool(reference_sql.strip())):
            if not active_tables:
                st.warning('Load an uploaded database before running a single-question result check.')
            else:
                try:
                    reference, _, reference_truncated = safe_query(reference_sql)
                    if reference_truncated:
                        st.warning('Reference results were truncated; comparison is not conclusive.')
                    comparison_rows = []
                    for kind in ('base', 'finetuned'):
                        try:
                            response = requests.post(
                                endpoint.rstrip('/') + '/generate-sql',
                                json={'question': question, 'schema': schema, 'model': kind},
                                timeout=90,
                            )
                            response.raise_for_status()
                            payload = response.json()
                            generated_sql = payload.get('extracted_sql', payload.get('sql'))
                            actual, _, truncated = safe_query(generated_sql)
                            comparison_rows.append({
                                'model': kind,
                                'parse_valid': True,
                                'execution_success': True,
                                'result_match': (
                                    False if truncated or reference_truncated else
                                    compare_results(reference, actual, order_sensitive=True)
                                ),
                                'error': 'truncated' if truncated else None,
                            })
                        except Exception as exc:
                            comparison_rows.append({
                                'model': kind,
                                'parse_valid': False,
                                'execution_success': False,
                                'result_match': False,
                                'error': str(exc),
                            })
                    st.caption('Single-question check only; this is not benchmark accuracy.')
                    st.session_state.comparison_rows = comparison_rows
                except Exception as exc:
                    st.error(f'Reference query failed: {exc}')
    if st.session_state.comparison_rows:
        st.caption('Saved single-question check')
        st.dataframe(
            pd.DataFrame(st.session_state.comparison_rows),
            hide_index=True,
            use_container_width=True,
        )
    if not endpoint:st.warning('Connect your inference backend to enable a real comparison.')
