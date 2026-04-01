"""Tools for SQL execution and schema exploration."""

import re
from pathlib import Path

import framework.agent as agentFramework
import framework.database as database

SQL_VALID_PREFIX = "SQL_VALID:"


def runSql(query: str, previewRows: int = 20) -> str:
    """Execute a SQL query and return a preview of the results."""
    previewRows = max(1, previewRows)

    # Block dangerous statements
    if re.search(r"\b(delete|drop|truncate|alter|update)\b", query, flags=re.IGNORECASE):
        return f"{SQL_VALID_PREFIX}ERROR\nSafety error: write statements are not allowed."

    validation = database.validate_query(query)
    if not validation.is_valid:
        return f"{SQL_VALID_PREFIX}ERROR\nSyntax error: {validation.error_message}"

    result = database.execute_query(query)
    if not result.is_success:
        return f"{SQL_VALID_PREFIX}ERROR\nExecution error: {result.error_message}"

    df = result.dataframe
    if df is None:
        return f"{SQL_VALID_PREFIX}ERROR\nQuery returned no dataframe."

    preview = df.head(previewRows)
    lines = [
        f"{SQL_VALID_PREFIX}OK",
        f"rowCount={df.height}",
        f"columnCount={df.width}",
        f"columns={', '.join(df.columns)}",
        "preview:",
        preview.write_csv(),
    ]
    return "\n".join(lines)


def listTables(schemaName: str) -> str:
    """List all tables in a schema."""
    tables = database.list_tables(schemaName)
    if not tables:
        return f"No tables found for schema '{schemaName}'. Check the schema name."
    return f"Tables in {schemaName}:\n- " + "\n- ".join(tables)


def describeTable(schemaName: str, tableName: str) -> str:
    """Describe the columns of a table."""
    columns = database.describe_table(schemaName, tableName)
    if not columns:
        return f"No columns found for {schemaName}.{tableName}. Check schema and table names."
    return f"Columns in {schemaName}.{tableName}:\n- " + "\n- ".join(columns)


RUN_SQL = agentFramework.Tool(
    name="run_sql",
    description=(
        "Execute a SQL query against the DuckDB database to validate and preview results. "
        "Use this to test your query before calling submit_answer. "
        "Returns row count, column names, and a data preview."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "SQL query to execute. Use schema-qualified table names (e.g. schema.table).",
            },
            "previewRows": {
                "type": "integer",
                "description": "Number of rows to preview (default 20).",
                "default": 20,
            },
        },
        "required": ["query"],
    },
    function=runSql,
)

LIST_TABLES = agentFramework.Tool(
    name="list_tables",
    description="List all tables in a given schema.",
    parameters={
        "type": "object",
        "properties": {
            "schemaName": {
                "type": "string",
                "description": "The schema name (e.g. 'Airline', 'financial').",
            },
        },
        "required": ["schemaName"],
    },
    function=listTables,
)

DESCRIBE_TABLE = agentFramework.Tool(
    name="describe_table",
    description=(
        "Show column names and types for a table. "
        "Use this before writing SQL to confirm exact column names."
    ),
    parameters={
        "type": "object",
        "properties": {
            "schemaName": {
                "type": "string",
                "description": "The schema name.",
            },
            "tableName": {
                "type": "string",
                "description": "The table name.",
            },
        },
        "required": ["schemaName", "tableName"],
    },
    function=describeTable,
)

# Keep VALIDATE_SQL_BUNDLE defined so evaluate.py import doesn't break.
# It's a thin wrapper around run_sql.
VALIDATE_SQL_BUNDLE = agentFramework.Tool(
    name="validate_sql_bundle",
    description=(
        "Validate a SQL query: checks syntax, executes it, and previews results. "
        "Use run_sql instead — this is kept for backwards compatibility."
    ),
    parameters={
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "The question being answered (unused, kept for compatibility).",
            },
            "query": {
                "type": "string",
                "description": "SQL query to validate.",
            },
        },
        "required": ["question", "query"],
    },
    function=lambda question="", query="", **_: runSql(query),
)
