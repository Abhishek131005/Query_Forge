"""Read-only SQLite query execution for local QueryForge workspaces.
Not a multi-tenant security boundary; deploy only with trusted local data.
"""
import sqlite3
import time
from pathlib import Path
import pandas as pd
import sqlglot
from sqlglot import exp

MAX_ROWS = 500
MAX_SECONDS = 5


def validate_sql(sql: str) -> str:
    if not sql or not sql.strip():
        raise ValueError('SQL query is empty.')
    try:
        statements = sqlglot.parse(sql, read='sqlite')
    except sqlglot.errors.ParseError as exc:
        raise ValueError(f'Invalid SQL: {exc}') from exc
    if len(statements) != 1 or statements[0] is None:
        raise ValueError('Exactly one SQL statement is allowed.')
    tree = statements[0]
    if not isinstance(tree, exp.Select):
        raise ValueError('Only SELECT queries are supported (no PRAGMA, EXPLAIN, or DDL).')
    forbidden = (exp.Insert, exp.Update, exp.Delete, exp.Create, exp.Drop, exp.Alter, exp.Command, exp.Attach, exp.Detach)
    if any(tree.find(node) is not None for node in forbidden):
        raise ValueError('Write operations and commands are prohibited.')
    return tree.sql(dialect='sqlite')


def execute_sql(sql: str, db_path: Path):
    validated = validate_sql(sql)
    if not db_path.is_file():
        raise ValueError('Workspace database does not exist. Upload a CSV first.')
    uri = db_path.resolve().as_uri() + '?mode=ro'
    start = time.monotonic()
    con = sqlite3.connect(uri, uri=True, timeout=2)
    try:
        con.execute('PRAGMA query_only=ON')
        con.set_authorizer(_authorizer)
        def progress():
            return 1 if time.monotonic() - start > MAX_SECONDS else 0
        con.set_progress_handler(progress, 1000)
        cur = con.execute(f'SELECT * FROM ({validated}) AS queryforge_result LIMIT {MAX_ROWS + 1}')
        names = [item[0] for item in cur.description]
        rows = cur.fetchmany(MAX_ROWS + 1)
        truncated = len(rows) > MAX_ROWS
        df = pd.DataFrame(rows[:MAX_ROWS], columns=names)
        return df, round(time.monotonic() - start, 3), truncated
    except sqlite3.DatabaseError as exc:
        raise ValueError(f'Query could not be executed: {exc}') from exc
    finally:
        con.close()


def _authorizer(action, arg1, arg2, dbname, source):
    allowed = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION, sqlite3.SQLITE_RECURSIVE}
    return sqlite3.SQLITE_OK if action in allowed else sqlite3.SQLITE_DENY
