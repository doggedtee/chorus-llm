import sqlite3
import time
import os
from typing import Optional
from pydantic import BaseModel
import anthropic

DB_PATH = os.getenv("RESEARCH_DB_PATH", "./data/research.db")

TABLE_SCHEMA = """
Table: papers
Columns:
  - id (INTEGER)
  - title (TEXT)
  - authors (TEXT)
  - year (INTEGER)
  - topic (TEXT)         -- values: 'climate change', 'machine learning'
  - citations (INTEGER)
  - abstract (TEXT)
"""

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


class SQLLookupResult(BaseModel):
    question: str
    sql_query: str
    rows: list
    columns: list[str]
    row_count: int
    latency_ms: float
    failure_mode: Optional[str] = None     # none, timeout, empty, malformed


def _nl_to_sql(question: str) -> str:
    """Use Claude to convert a natural language question to a SQL query."""
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=256,
        messages=[{
            "role": "user",
            "content": (
                f"Convert this question to a SQLite SQL query.\n"
                f"Schema:\n{TABLE_SCHEMA}\n"
                f"Question: {question}\n"
                f"Return ONLY the SQL query, nothing else."
            ),
        }],
    )
    return response.content[0].text.strip()


def sql_lookup(question: str, timeout: float = 10.0) -> SQLLookupResult:
    """
    Converts a natural language question to SQL and queries the research database.

    Failure contracts:
    - malformed: question is blank or not a string
    - empty:     query runs but returns no rows
    - timeout:   DB query takes too long
    """
    start = time.time()

    # malformed input
    if not question or not isinstance(question, str):
        return SQLLookupResult(
            question=str(question),
            sql_query="",
            rows=[],
            columns=[],
            row_count=0,
            latency_ms=0.0,
            failure_mode="malformed",
        )

    try:
        sql_query = _nl_to_sql(question)

        conn = sqlite3.connect(DB_PATH, timeout=timeout)
        cursor = conn.cursor()
        cursor.execute(sql_query)

        columns = [desc[0] for desc in cursor.description] if cursor.description else []
        rows = cursor.fetchall()
        conn.close()

        latency = (time.time() - start) * 1000
        print(f"[sql_lookup] rows={len(rows)} latency={latency:.0f}ms sql='{sql_query}'")

        if not rows:
            return SQLLookupResult(
                question=question,
                sql_query=sql_query,
                rows=[],
                columns=columns,
                row_count=0,
                latency_ms=latency,
                failure_mode="empty",
            )

        return SQLLookupResult(
            question=question,
            sql_query=sql_query,
            rows=[list(row) for row in rows],
            columns=columns,
            row_count=len(rows),
            latency_ms=latency,
            failure_mode="none",
        )

    except Exception as e:
        latency = (time.time() - start) * 1000
        return SQLLookupResult(
            question=question,
            sql_query="",
            rows=[],
            columns=[],
            row_count=0,
            latency_ms=latency,
            failure_mode="malformed",
        )