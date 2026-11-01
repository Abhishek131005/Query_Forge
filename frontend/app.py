import os
import re
import sqlite3
import requests
import pandas as pd
import plotly.express as px
from database import import_csv, tables, schema_sql, table_info, delete_table, clean_identifier, connect
import streamlit as st

st.set_page_config(page_title='QueryForge AI', page_icon='⚡', layout='wide')
st.markdown('''<style>
.stApp {background:#0b1020;color:#e9ecf7} [data-testid="stSidebar"]{background:#11182c}
.block-container{padding-top:2rem;max-width:1400px} h1,h2,h3{letter-spacing:-.035em}
.hero{padding:26px;border:1px solid #293451;border-radius:20px;background:linear-gradient(120deg,#192448,#17142d)}
.small{color:#aeb9d2} .stButton button{border-radius:10px}
</style>''', unsafe_allow_html=True)

@st.cache_resource
def demo_db():
    conn=sqlite3.connect(':memory:',check_same_thread=False)
    conn.execute('CREATE TABLE employees (id INTEGER PRIMARY KEY, name TEXT, department TEXT, salary INTEGER)')
    conn.executemany('INSERT INTO employees VALUES (?,?,?,?)',[(1,'Asha','Engineering',120000),(2,'Rahul','Sales',75000),(3,'Mira','Engineering',145000),(4,'Neha','Finance',92000),(5,'Kabir','Engineering',85000)])
    conn.commit()
    return conn

if 'history' not in st.session_state: st.session_state.history=[]
with st.sidebar:
    st.title('⚡ QueryForge AI')
    page=st.radio('Workspace',['AI SQL Studio','Database Explorer','Model Comparison','Model Intelligence'])
    st.divider()
    st.caption('Inference endpoint')
    endpoint=st.text_input('Backend URL',value=os.getenv('INFERENCE_API_URL',''),placeholder='https://your-api.example.com',help='Leave blank to explore the UI in demo mode.')
    st.caption('Demo mode uses sample SQL, not the fine-tuned model.')

# Demo database is used only until a user imports a CSV.
active_tables = tables()
conn = None if active_tables else demo_db()
schema = schema_sql() if active_tables else 'CREATE TABLE employees (id INTEGER PRIMARY KEY, name TEXT, department TEXT, salary INTEGER);'
with st.sidebar:
    st.divider()
    st.caption(f'Active database: {len(active_tables)} uploaded table(s)' if active_tables else 'Active database: demo employees')

def generate_sql(question,schema,model='finetuned'):
    if not endpoint:
        q=question.lower()
        if 'department' in q and ('count' in q or 'how many' in q):return 'SELECT department, COUNT(*) AS employee_count FROM employees GROUP BY department;',True
        if 'average' in q:return 'SELECT department, AVG(salary) AS average_salary FROM employees GROUP BY department;',True
        if 'engineering' in q:return "SELECT name, salary FROM employees WHERE department = 'Engineering' AND salary > 100000;",True
        return 'SELECT * FROM employees LIMIT 10;',True
    response=requests.post(endpoint.rstrip('/')+'/generate-sql',json={'question':question,'schema':schema,'model':model},timeout=90)
    response.raise_for_status()
    return response.json()['sql'],False

def safe_query(sql):
    if active_tables:
        raise ValueError('Execution against uploaded datasets is reserved for Milestone 3. SQL generation works now.')
    s=sql.strip().rstrip(';').strip()
    if ';' in s or not re.match(r'^SELECT\b',s,re.I):raise ValueError('Only one read-only SELECT query is allowed in this demo.')
    if re.search(r'\b(ATTACH|DETACH|PRAGMA|INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|REPLACE|VACUUM)\b',s,re.I):raise ValueError('Potentially unsafe SQL rejected.')
    return pd.read_sql_query('SELECT * FROM ('+s+') LIMIT 200',conn)

if page=='AI SQL Studio':
    st.markdown('<div class="hero"><h1>Ask your data anything.</h1><p class="small">Turn natural language into SQL with a LoRA fine-tuned language model.</p></div>',unsafe_allow_html=True)
    st.write('')
    a,b=st.columns([1,1])
    with a:
        st.subheader('Your question')
        question=st.text_area('Ask in plain English',value='List Engineering employees earning more than 100000.',height=125)
        user_schema=st.text_area('Database schema (auto-generated from your CSV tables)',value=schema,height=145)
        run=st.button('⚡ Generate SQL',type='primary',use_container_width=True)
    with b:
        st.subheader('Generated SQL')
        if run:
            try:
                sql,demo=generate_sql(question,user_schema)
                st.session_state.sql=sql
                st.session_state.is_demo=demo
                st.session_state.history.append({'question':question,'sql':sql})
            except Exception as e:st.error(f'Inference failed: {e}')
        sql=st.text_area('SQL editor',value=st.session_state.get('sql',''),height=190)
        if st.session_state.get('is_demo',not endpoint):st.info('Demo response. Connect a model backend for real inference.')
        execute=st.button('▶ Run read-only query',use_container_width=True)
    if execute and sql:
        try:
            df=safe_query(sql)
            st.subheader('Query results')
            st.dataframe(df,use_container_width=True,hide_index=True)
            st.download_button('Download CSV',df.to_csv(index=False),'query_results.csv','text/csv')
            numeric=df.select_dtypes(include='number').columns.tolist()
            if len(df.columns)>1 and numeric:
                st.plotly_chart(px.bar(df,x=df.columns[0],y=numeric[0],template='plotly_dark'),use_container_width=True)
        except Exception as e:st.error(str(e))
    if st.session_state.history:
        with st.expander('Recent generations'):
            for item in reversed(st.session_state.history[-5:]):st.code(item['sql'],language='sql')
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
            st.rerun()
        st.info('Go to AI SQL Studio to generate SQL against these uploaded tables. Executing generated SQL on them is a Milestone 3 feature.')
    else:
        st.info('No uploaded tables yet. The AI SQL Studio currently uses its built-in employees demo schema.')
        st.code(schema, language='sql')
        st.dataframe(pd.read_sql_query('SELECT * FROM employees', demo_db()), use_container_width=True, hide_index=True)
elif page=='Model Comparison':
    st.title('Base vs Fine-tuned')
    st.caption('Requires an API that supports model="base" and model="finetuned". No fabricated comparison metrics.')
    question=st.text_input('Question','List Engineering employees earning more than 100000.')
    if st.button('Compare models',disabled=not bool(endpoint)):
        cols=st.columns(2)
        for col,kind in zip(cols,['base','finetuned']):
            with col:
                st.subheader(kind.title())
                try:st.code(generate_sql(question,schema,kind)[0],language='sql')
                except Exception as e:st.error(str(e))
    if not endpoint:st.warning('Connect your inference backend to enable a real comparison.')
else:
    st.title('Model Intelligence')
    st.caption('Configuration extracted from your training notebook. Not measured accuracy.')
    c=st.columns(4)
    for col,(label,value) in zip(c,[('Base model','TinyLlama 1.1B'),('LoRA rank','16'),('Training examples','2,850'),('Epochs','1')]):col.metric(label,value)
    st.markdown('**Dataset:** b-mc2/sql-create-context (3,000 selected; 95% training / 5% evaluation)')
    st.markdown('**LoRA targets:** q_proj, k_proj, v_proj, o_proj')
    st.info('Add real evaluation scores and training-loss logs after measuring them. SQL formatting alone is not proof of correctness.')
