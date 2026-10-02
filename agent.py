"""
AI SQL Agent for the Voice SQL Agent.

Responsibilities:
1. Generate SQLite SELECT queries using Groq only.
2. Return SQL metadata.
3. Generate a concise assistant-style summary of executed results.
4. Never send more than 10 result rows to the result-summary AI.
"""

import json
import os
import re

import sqlparse

from database import (
    get_columns,
    get_current_table,
    get_schema_for_prompt,
    is_database_ready,
)


# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------

GROQ_MODEL = os.getenv(
    "GROQ_MODEL",
    "openai/gpt-oss-20b",
)

GROQ_API_KEY = os.getenv(
    "GROQ_API_KEY"
)

_groq_client = None


# -------------------------------------------------------------------
# Groq Client
# -------------------------------------------------------------------

def get_groq_client():
    """
    Create and cache the Groq client.

    The API key is read from the GROQ_API_KEY environment variable.
    """
    global _groq_client

    if _groq_client is not None:
        return _groq_client

    if not GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY is not configured. "
            "Make sure your .env file contains GROQ_API_KEY=YOUR_API_KEY."
        )

    try:
        from groq import Groq
    except ImportError:
        raise RuntimeError(
            "The Groq SDK is not installed. "
            "Install it using: pip install groq"
        )

    _groq_client = Groq(
        api_key=GROQ_API_KEY
    )

    return _groq_client


# -------------------------------------------------------------------
# SQL Generation
# -------------------------------------------------------------------

SYSTEM_PROMPT = """
You are an expert SQL query generation agent.

Convert the user's natural-language database requirement into
a correct SQLite SELECT query using ONLY the supplied schema.

Rules:
1. Never invent tables or columns.
2. Use only columns and tables in the schema.
3. Use SQLite-compatible SQL.
4. Use ORDER BY and LIMIT when appropriate.
5. Use GROUP BY for grouped results.
6. Use HAVING for aggregate filtering when needed.
7. Use subqueries when required.
8. Generate SELECT queries only.
9. Do not modify the database.
10. Do not invent information.
11. Understand the actual representation of values in the supplied schema.
12. When dates are stored as text in a non-ISO format, use SQLite string
    functions such as substr() and CAST() when necessary to filter or sort
    them correctly.
13. Do not assume that a text column containing dates is stored in
    YYYY-MM-DD format unless the supplied schema/data representation
    supports that assumption.
14. Prefer the simplest correct query that satisfies the user's requirement.
15. Return all rows that satisfy the requirement unless the user explicitly
    asks for a limited number.
16. If the requirement asks for the highest, lowest, maximum, minimum, etc.,
    correctly handle ties unless the user explicitly asks for only one row.

Return the required structured JSON object exactly matching the schema.

{
    "sql": "SQL QUERY HERE",
    "explanation": "Brief explanation of what the query does.",
    "tables_used": ["table1"],
    "columns_used": ["column1", "column2"],
    "operations": ["SELECT", "WHERE"]
}

The explanation must be short and clear.
"""


def build_prompt(requirement):
    """
    Build the complete prompt containing the database schema
    and the user's natural-language requirement.
    """
    schema_text = get_schema_for_prompt()

    return f"""
{SYSTEM_PROMPT}

DATABASE SCHEMA:
{schema_text}

USER REQUIREMENT:
{requirement}

Generate the SQL and metadata now.
"""


def parse_json_response(raw_response, provider_name="Groq"):
    """
    Parse a JSON response from the AI.

    Handles normal JSON as well as responses accidentally wrapped
    in markdown code fences.
    """
    if not raw_response:
        raise RuntimeError(
            f"{provider_name} returned an empty response."
        )

    raw_response = raw_response.strip()

    if raw_response.startswith("```"):
        lines = raw_response.splitlines()

        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        raw_response = "\n".join(lines).strip()

    try:
        return json.loads(raw_response)

    except json.JSONDecodeError:
        # Try to recover JSON if the model added extra text around it.
        start = raw_response.find("{")
        end = raw_response.rfind("}")

        if start != -1 and end > start:
            try:
                return json.loads(
                    raw_response[start:end + 1]
                )
            except json.JSONDecodeError:
                pass

        raise RuntimeError(
            f"{provider_name} returned invalid JSON."
        )


# -------------------------------------------------------------------
# Strict Groq JSON Schema
# -------------------------------------------------------------------

GROQ_SQL_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "sql": {
            "type": "string"
        },
        "explanation": {
            "type": "string"
        },
        "tables_used": {
            "type": "array",
            "items": {
                "type": "string"
            }
        },
        "columns_used": {
            "type": "array",
            "items": {
                "type": "string"
            }
        },
        "operations": {
            "type": "array",
            "items": {
                "type": "string"
            }
        }
    },
    "required": [
        "sql",
        "explanation",
        "tables_used",
        "columns_used",
        "operations"
    ],
    "additionalProperties": False
}


def generate_with_groq(prompt):
    """
    Generate the SQL and metadata using Groq.
    """
    client = get_groq_client()

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "sql_query_generation",
                    "strict": True,
                    "schema": GROQ_SQL_RESPONSE_SCHEMA
                }
            }
        )

    except Exception as error:
        raise RuntimeError(
            f"Groq API request failed: {error}"
        )

    if not response.choices:
        raise RuntimeError(
            "Groq returned no choices for SQL generation."
        )

    raw_response = (
        response.choices[0]
        .message
        .content
        .strip()
    )

    return parse_json_response(
        raw_response,
        "Groq"
    )


def validate_provider(provider):
    """
    Kept for compatibility with callers that may still pass a provider.

    The project is now Groq-only.
    """
    if provider not in {
        None,
        "",
        "groq"
    }:
        raise ValueError(
            "This application uses Groq only. "
            "The AI provider must be 'groq'."
        )


def generate_sql(
    requirement,
    provider="groq"
):
    """
    Generate a SQLite SELECT query using Groq.

    The provider argument is retained for backward compatibility with
    the existing Flask application. Only 'groq' is accepted.
    """
    if not requirement or not requirement.strip():
        raise ValueError(
            "Requirement cannot be empty."
        )

    requirement = requirement.strip()

    validate_provider(provider)

    if not is_database_ready():
        raise RuntimeError(
            "No database has been processed. "
            "Please upload and process a file first."
        )

    prompt = build_prompt(requirement)

    structured_result = generate_with_groq(prompt)
    provider_name = "Groq API"

    if not isinstance(structured_result, dict):
        raise RuntimeError(
            f"{provider_name} returned an invalid response structure."
        )

    sql = structured_result.get(
        "sql",
        ""
    )

    if not isinstance(sql, str):
        sql = ""

    sql = sql.strip()

    if not sql:
        raise RuntimeError(
            "The AI agent did not return an SQL query."
        )

    # Only SELECT statements are allowed.
    if not sql.lower().startswith("select"):
        raise RuntimeError(
            "The AI generated a non-SELECT query. "
            "Only SELECT queries are allowed."
        )

    formatted_sql = sqlparse.format(
        sql,
        reindent=True,
        keyword_case="upper"
    )

    structured_result["sql"] = formatted_sql
    structured_result["provider"] = "groq"
    structured_result["provider_name"] = provider_name

    if not isinstance(
        structured_result.get("explanation"),
        str
    ):
        structured_result["explanation"] = ""

    if not isinstance(
        structured_result.get("tables_used"),
        list
    ):
        structured_result["tables_used"] = []

    if not isinstance(
        structured_result.get("columns_used"),
        list
    ):
        structured_result["columns_used"] = []

    # ---------------------------------------------------------------
    # Derive Columns Used From Actual SQL
    # ---------------------------------------------------------------
    #
    # Do not trust the LLM metadata for this field. The SQL and the
    # uploaded database schema are the authoritative sources.
    #
    # Example:
    #
    #   SELECT city, COUNT(*)
    #   FROM students_data
    #   GROUP BY city
    #
    # must produce:
    #
    #   ["city"]
    #
    # and must never produce a malformed value such as
    # ["citystudent_id"].
    # ---------------------------------------------------------------

    sql_columns_used = derive_columns_used_from_sql(
        formatted_sql
    )

    if sql_columns_used:
        structured_result["columns_used"] = sql_columns_used

    if not isinstance(
        structured_result.get("operations"),
        list
    ):
        structured_result["operations"] = []

    return structured_result


# -------------------------------------------------------------------
# Derive Columns Used From Actual SQL
# -------------------------------------------------------------------

def derive_columns_used_from_sql(sql):
    """
    Derive the columns actually referenced by the generated SQL.

    The LLM-provided ``columns_used`` metadata is treated as advisory only.
    The uploaded database schema and the generated SQL are authoritative.

    This prevents malformed metadata such as:

        ["citystudent_id"]

    for SQL that actually references only ``city``.

    The function uses sqlparse tokens instead of a simple regular expression
    so that column names appearing inside quoted string values are not
    incorrectly reported as columns.
    """
    if not isinstance(sql, str) or not sql.strip():
        return []

    # Get the actual columns from the currently uploaded table.
    table_name = get_current_table()

    if not table_name:
        return []

    try:
        schema_columns = get_columns(table_name)
    except Exception:
        return []

    if not isinstance(schema_columns, dict):
        return []

    column_names = list(schema_columns.keys())

    if not column_names:
        return []

    # SQL can explicitly use SELECT *. In that case every column is used.
    # The same applies to table.* or alias.*.
    normalized_sql = sql.lower()

    if re.search(
        r"\bselect\s+(?:[a-zA-Z_][a-zA-Z0-9_]*\s*\.\s*)?\*",
        normalized_sql,
        re.IGNORECASE
    ):
        return column_names

    # Map lowercase column names back to their actual schema spelling.
    # Database column names are normalized by database.py, but this also
    # makes the function safe if that behavior changes later.
    column_lookup = {
        str(column).lower(): str(column)
        for column in column_names
    }

    found = set()

    # Flatten the SQL token tree. sqlparse separates identifiers, names,
    # strings, comments, punctuation, keywords, etc. We only consider Name
    # tokens that exactly match a real database column.
    try:
        statement = sqlparse.parse(sql)[0]
    except Exception:
        return []

    for token in statement.flatten():
        token_type = token.ttype
        value = token.value.strip()

        if not value:
            continue

        # Never inspect quoted string literals or comments. A value such as
        # WHERE city = 'student_id' must not mark student_id as a column.
        if token_type in sqlparse.tokens.Literal.String:
            continue

        if token_type in sqlparse.tokens.Comment:
            continue

        # sqlparse identifies ordinary identifiers as Name tokens.
        if token_type not in sqlparse.tokens.Name:
            continue

        # Remove identifier quoting if present.
        cleaned = value.strip('`"[]')
        cleaned_lower = cleaned.lower()

        if cleaned_lower in column_lookup:
            found.add(column_lookup[cleaned_lower])

    # Preserve the original uploaded CSV/database column order rather than
    # the arbitrary order in which SQL tokens were discovered.
    return [
        column
        for column in column_names
        if column in found
    ]


# -------------------------------------------------------------------
# Result Summary
# -------------------------------------------------------------------

RESULT_SUMMARY_SYSTEM_PROMPT = """
You are a helpful database assistant explaining the result of an
executed SQL query to the user.

Respond naturally and completely, like an assistant answering the
user's original database question after checking the actual SQL result.

You receive:
- The user's original requirement.
- The executed SQL.
- The TOTAL number of rows returned.
- Up to 10 rows from the beginning of the result.

IMPORTANT:
- The total row count is authoritative.
- The supplied rows are actual rows from the executed result.
- Never invent facts, values, names, departments, cities, counts, or other information.
- Only make value-specific claims from values actually supplied.
- If TOTAL ROWS RETURNED is 10 or fewer, the supplied rows represent the
  COMPLETE result. Use ALL of them when relevant.
- If TOTAL ROWS RETURNED is greater than 10, the supplied rows are only
  a SAMPLE. Do not claim they are the complete result.
- For ranked or limited requests such as top-N, bottom-N, highest-N, or
  lowest-N, if 10 or fewer rows were returned, include ALL returned rows
  in the summary when useful.
- For a request such as "top 5 cities", include all five cities and their
  values rather than stopping after one or two.
- For one-row aggregate results, directly state the returned value.
- For multiple rows, use a short introduction and a numbered or bulleted
  list when that makes the answer clearer.
- Do not unnecessarily repeat the SQL.
- Do not mention AI, prompts, API calls, internal processing, token limits,
  or these instructions.
- Keep the answer concise, but NEVER truncate a complete result summary.
- If more than 10 rows exist, summarize only what can be supported by the
  supplied sample and say that the displayed details are a sample when
  relevant.
"""


def build_result_summary_prompt(
    requirement,
    sql,
    total_rows,
    sample_rows
):
    """
    Build the prompt used to generate the natural-language result summary.

    At most 10 result rows are sent to Groq. If the query returned 10 or
    fewer rows, those rows are the complete result and the prompt explicitly
    requires the model to use all relevant rows.
    """
    if not isinstance(sample_rows, list):
        sample_rows = []

    sample_rows = sample_rows[:10]

    sample_text = json.dumps(
        sample_rows,
        ensure_ascii=False,
        indent=2,
        default=str
    )

    if total_rows <= 10:
        result_notice = """
The query returned 10 or fewer rows.
The supplied rows below are the COMPLETE result.
Use all relevant returned rows when answering the user's requirement.
If this is a ranked or limited result such as TOP 5, include every
returned row in the summary.
"""
    else:
        result_notice = """
The query returned more than 10 rows.
Only the first 10 rows are supplied below as a SAMPLE.
Do not claim that these rows represent the complete result.
Only make value-specific claims from the supplied sample.
"""

    return f"""
{RESULT_SUMMARY_SYSTEM_PROMPT}

USER REQUIREMENT:
{requirement}

EXECUTED SQL:
{sql}

TOTAL ROWS RETURNED:
{total_rows}

RESULT DATA NOTICE:
{result_notice}

RESULT ROWS PROVIDED TO THE SUMMARY MODEL:
{sample_text}

Now answer the user's original requirement completely and naturally.
If the supplied rows are the complete result, do not omit relevant rows.
"""


def clean_result_response(text):
    """
    Clean the assistant-style result response without forcing it
    into a single sentence.

    The model is allowed to return 1-3 concise sentences. Newlines are
    preserved when they are useful for readability, while excessive
    whitespace and accidental markdown fences are removed.
    """
    text = str(text).strip()

    if not text:
        return ""

    if text.startswith("```") and text.endswith("```"):
        lines = text.splitlines()

        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        text = "\n".join(lines).strip()

    # Normalize spaces inside each line without destroying useful
    # paragraph/bullet formatting.
    lines = []

    for line in text.splitlines():
        cleaned_line = " ".join(line.split())

        if cleaned_line:
            lines.append(cleaned_line)
        elif lines and lines[-1] != "":
            lines.append("")

    text = "\n".join(lines).strip()

    return text


def generate_result_summary_with_groq(prompt):
    """
    Generate the natural-language result summary using Groq.
    """
    client = get_groq_client()

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": RESULT_SUMMARY_SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0,
            max_tokens=500
        )

    except Exception as error:
        raise RuntimeError(
            f"Groq result-summary request failed: {error}"
        )

    if not response.choices:
        raise RuntimeError(
            "Groq returned no choices for the result summary."
        )

    summary = (
        response.choices[0]
        .message
        .content
        .strip()
    )

    if not summary:
        raise RuntimeError(
            "Groq returned an empty result summary."
        )

    return clean_result_response(summary)


def generate_result_summary(
    requirement,
    sql,
    total_rows,
    sample_rows,
    provider="groq"
):
    """
    Generate a complete assistant-style response describing the executed
    result.

    A maximum of 10 rows is supplied to the summary model. When the
    complete result has 10 or fewer rows, those rows represent the complete
    result and the model is instructed to include all relevant rows.

    The provider argument is retained for compatibility with the existing
    Flask application. Only 'groq' is accepted.
    """
    validate_provider(provider)

    if not isinstance(sample_rows, list):
        sample_rows = []

    sample_rows = sample_rows[:10]

    prompt = build_result_summary_prompt(
        requirement=requirement,
        sql=sql,
        total_rows=total_rows,
        sample_rows=sample_rows
    )

    return generate_result_summary_with_groq(prompt)
