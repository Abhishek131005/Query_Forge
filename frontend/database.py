"""Per-session SQLite CSV workspace for QueryForge AI."""
import io
import re
import sqlite3
import uuid
from pathlib import Path

import pandas as pd
import streamlit as st

DATA_DIR = Path(__file__).resolve().parent / '.local_data'
DATA_DIR.mkdir(exist_ok=True)
MAX_BYTES = 10 * 1024 * 1024
MAX_ROWS = 100_000
MAX_COLUMNS = 100


def database_path():
    if 'database_id' not in st.session_state:
        st.session_state.database_id = uuid.uuid4().hex
    return DATA_DIR / f"{st.session_state.database_id}.sqlite3"


def connect():
    return sqlite3.connect(str(database_path()))


def clean_identifier(value):
    value = re.sub(r'[^a-zA-Z0-9_]', '_', str(value).strip())
    value = re.sub(r'_+', '_', value).strip('_').lower()
    if not value:
        value = 'column'
    if value[0].isdigit():
        value = 'col_' + value
    return value[:60]


def unique_names(values):
    used = set()
    result = []
    for value in values:
        base = clean_identifier(value)
        candidate = base
        index = 2
        while candidate.lower() in used:
            candidate = f'{base[:50]}_{index}'
            index += 1
        used.add(candidate.lower())
        result.append(candidate)
    return result


def import_csv(uploaded, table_name):
    if uploaded.size > MAX_BYTES:
        raise ValueError('CSV exceeds 10 MB limit.')
    data = uploaded.getvalue()
    try:
        df = pd.read_csv(io.BytesIO(data), nrows=MAX_ROWS + 1, low_memory=False)
    except Exception as exc:
        raise ValueError(f'Could not parse CSV: {exc}') from exc
    if df.empty:
        raise ValueError('CSV has no data rows.')
    if len(df) > MAX_ROWS:
        raise ValueError('CSV exceeds 100,000 rows.')
    if len(df.columns) > MAX_COLUMNS:
        raise ValueError('CSV exceeds 100 columns.')
    df.columns = unique_names(df.columns)
    table = clean_identifier(table_name)
    if table.startswith('sqlite_'):
        raise ValueError('Table name cannot start with sqlite_.')
    with connect() as conn:
        df.to_sql(table, conn, if_exists='fail', index=False, chunksize=1000)
    return table, len(df), len(df.columns)


def tables():
    with connect() as conn:
        rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name").fetchall()
    return [row[0] for row in rows]


def quote_ident(name):
    return '"' + name.replace('"', '""') + '"'


def schema_sql():
    with connect() as conn:
        rows = conn.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name").fetchall()
    return '\n\n'.join(row[0] + ';' for row in rows if row[0])


def table_info(name):
    if name not in tables():
        raise ValueError('Unknown table')
    with connect() as conn:
        columns = pd.read_sql_query(f'PRAGMA table_info({quote_ident(name)})', conn)
        count = conn.execute(f'SELECT COUNT(*) FROM {quote_ident(name)}').fetchone()[0]
        preview = pd.read_sql_query(f'SELECT * FROM {quote_ident(name)} LIMIT 50', conn)
    return columns, count, preview


def delete_table(name):
    if name not in tables():
        raise ValueError('Unknown table')
    with connect() as conn:
        conn.execute(f'DROP TABLE {quote_ident(name)}')
