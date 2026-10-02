# database.py

"""
Dynamic Database Management for the Voice SQL Agent.

This module is responsible for:

1. Processing an uploaded CSV file.
2. Dynamically detecting its table structure.
3. Preserving the original CSV column order.
4. Creating a temporary SQLite database.
5. Importing the uploaded CSV data into SQLite.
6. Extracting the database schema dynamically.
7. Providing the schema to the AI SQL agent.
8. Validating generated SQL.
9. Executing SELECT queries.
10. Returning query results.

Only ONE table is supported per uploaded file.
"""


import sqlite3
import os
import re

import pandas as pd


# -------------------------------------------------------------------
# Database Configuration
# -------------------------------------------------------------------

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


DATABASE_PATH = os.path.join(
    BASE_DIR,
    "database.db"
)


# -------------------------------------------------------------------
# Current Table
# -------------------------------------------------------------------

CURRENT_TABLE_NAME = None


# -------------------------------------------------------------------
# Current Schema
# -------------------------------------------------------------------

DATABASE_SCHEMA = {}


# -------------------------------------------------------------------
# Current Column Order
# -------------------------------------------------------------------

CURRENT_COLUMN_ORDER = []


# -------------------------------------------------------------------
# Allowed File Extensions
# -------------------------------------------------------------------

ALLOWED_EXTENSIONS = {

    ".csv"

}


# -------------------------------------------------------------------
# Get Database Connection
# -------------------------------------------------------------------

def get_connection():
    """
    Create and return a connection to the SQLite database.

    Returns:
        sqlite3.Connection:
            SQLite database connection.
    """

    return sqlite3.connect(
        DATABASE_PATH
    )


# -------------------------------------------------------------------
# Validate Uploaded File
# -------------------------------------------------------------------

def validate_file(
    file_path
):
    """
    Validate the uploaded file.

    Currently only CSV files are supported.

    Args:
        file_path (str):
            Path of the uploaded file.

    Raises:
        ValueError:
            If the file type is unsupported.
    """

    if not file_path:

        raise ValueError(
            "No file was provided."
        )


    extension = os.path.splitext(
        file_path
    )[1].lower()


    if extension not in ALLOWED_EXTENSIONS:

        raise ValueError(
            "Unsupported file type. "
            "Only CSV files are currently supported."
        )


# -------------------------------------------------------------------
# Generate Safe Table Name
# -------------------------------------------------------------------

def generate_table_name(
    file_path
):
    """
    Generate a safe SQLite table name from the uploaded filename.

    Example:

        employees_data.csv
        ↓
        employees_data

    Args:
        file_path (str):
            Uploaded file path.

    Returns:
        str:
            Safe SQLite table name.
    """

    filename = os.path.basename(
        file_path
    )


    name = os.path.splitext(
        filename
    )[0]


    # ---------------------------------------------------------------
    # Replace invalid characters
    # ---------------------------------------------------------------

    name = re.sub(
        r"[^a-zA-Z0-9_]",
        "_",
        name
    )


    # ---------------------------------------------------------------
    # Convert multiple underscores
    # ---------------------------------------------------------------

    name = re.sub(
        r"_+",
        "_",
        name
    )


    # ---------------------------------------------------------------
    # Remove leading/trailing underscores
    # ---------------------------------------------------------------

    name = name.strip(
        "_"
    )


    # ---------------------------------------------------------------
    # Prevent empty table names
    # ---------------------------------------------------------------

    if not name:

        name = "uploaded_data"


    # ---------------------------------------------------------------
    # Prevent table names beginning with a number
    # ---------------------------------------------------------------

    if name[0].isdigit():

        name = "table_" + name


    return name.lower()


# -------------------------------------------------------------------
# Convert Pandas Data Type to SQLite Type
# -------------------------------------------------------------------

def pandas_type_to_sqlite(
    data_type
):
    """
    Convert a Pandas data type into a SQLite-compatible type.

    Args:
        data_type:
            Pandas column data type.

    Returns:
        str:
            SQLite data type.
    """

    if pd.api.types.is_integer_dtype(
        data_type
    ):

        return "INTEGER"


    if pd.api.types.is_float_dtype(
        data_type
    ):

        return "REAL"


    if pd.api.types.is_bool_dtype(
        data_type
    ):

        return "BOOLEAN"


    if pd.api.types.is_datetime64_any_dtype(
        data_type
    ):

        return "DATETIME"


    return "TEXT"


# -------------------------------------------------------------------
# Clean Column Names
# -------------------------------------------------------------------

def clean_column_names(
    dataframe
):
    """
    Clean column names while preserving their original order.

    Example:

        Employee Name
        ↓
        employee_name

    The position of every column remains unchanged.

    Args:
        dataframe:
            Pandas DataFrame.

    Returns:
        pandas.DataFrame:
            DataFrame with cleaned column names.
    """

    cleaned_columns = []


    # ---------------------------------------------------------------
    # Process Columns In Their Existing Order
    # ---------------------------------------------------------------

    for column in dataframe.columns:

        column_name = str(
            column
        ).strip()


        # -----------------------------------------------------------
        # Replace spaces and special characters
        # -----------------------------------------------------------

        column_name = re.sub(
            r"[^a-zA-Z0-9_]",
            "_",
            column_name
        )


        # -----------------------------------------------------------
        # Convert multiple underscores
        # -----------------------------------------------------------

        column_name = re.sub(
            r"_+",
            "_",
            column_name
        )


        # -----------------------------------------------------------
        # Remove leading/trailing underscores
        # -----------------------------------------------------------

        column_name = column_name.strip(
            "_"
        )


        # -----------------------------------------------------------
        # Prevent empty column name
        # -----------------------------------------------------------

        if not column_name:

            column_name = "column"


        # -----------------------------------------------------------
        # Prevent column name starting with number
        # -----------------------------------------------------------

        if column_name[0].isdigit():

            column_name = "column_" + column_name


        cleaned_columns.append(
            column_name.lower()
        )


    # ---------------------------------------------------------------
    # Ensure Column Names Are Unique
    # ---------------------------------------------------------------

    unique_columns = []

    used_names = {}


    for column_name in cleaned_columns:

        if column_name not in used_names:

            used_names[
                column_name
            ] = 1

            unique_columns.append(
                column_name
            )

        else:

            used_names[
                column_name
            ] += 1


            new_name = (

                f"{column_name}_"
                f"{used_names[column_name]}"

            )


            unique_columns.append(
                new_name
            )


    # ---------------------------------------------------------------
    # Apply Cleaned Names
    # ---------------------------------------------------------------

    dataframe.columns = unique_columns


    return dataframe


# -------------------------------------------------------------------
# Reset Previous Database
# -------------------------------------------------------------------

def reset_database():
    """
    Remove the previous temporary database.

    This allows a new uploaded file to replace
    the previously processed dataset.
    """

    global CURRENT_TABLE_NAME
    global DATABASE_SCHEMA
    global CURRENT_COLUMN_ORDER


    # ---------------------------------------------------------------
    # Close Existing Connection
    # ---------------------------------------------------------------

    try:

        connection = get_connection()

        connection.close()

    except Exception:

        pass


    # ---------------------------------------------------------------
    # Remove Database File
    # ---------------------------------------------------------------

    if os.path.exists(
        DATABASE_PATH
    ):

        try:

            os.remove(
                DATABASE_PATH
            )

        except PermissionError:

            raise RuntimeError(

                "Could not replace the current database. "
                "Please try again."

            )


    # ---------------------------------------------------------------
    # Reset Current State
    # ---------------------------------------------------------------

    CURRENT_TABLE_NAME = None

    DATABASE_SCHEMA = {}

    CURRENT_COLUMN_ORDER = []


# -------------------------------------------------------------------
# Process Uploaded CSV
# -------------------------------------------------------------------

def process_uploaded_file(
    file_path
):
    """
    Process an uploaded CSV file.

    Steps:

    1. Validate file.
    2. Read CSV.
    3. Capture original column order.
    4. Clean column names.
    5. Preserve column order.
    6. Create SQLite database.
    7. Create one table.
    8. Insert CSV data.
    9. Extract schema in the same order.

    Args:
        file_path (str):
            Path of uploaded CSV file.

    Returns:
        dict:
            Information about the processed database.
    """

    global CURRENT_TABLE_NAME
    global DATABASE_SCHEMA
    global CURRENT_COLUMN_ORDER


    # ---------------------------------------------------------------
    # Validate File
    # ---------------------------------------------------------------

    validate_file(
        file_path
    )


    # ---------------------------------------------------------------
    # Read CSV
    # ---------------------------------------------------------------

    try:

        dataframe = pd.read_csv(
            file_path
        )

    except Exception as error:

        raise RuntimeError(

            f"Could not read CSV file: {error}"

        )


    # ---------------------------------------------------------------
    # Validate Data
    # ---------------------------------------------------------------

    if dataframe.empty:

        raise ValueError(
            "The uploaded CSV file is empty."
        )


    if len(
        dataframe.columns
    ) == 0:

        raise ValueError(
            "The uploaded CSV file has no columns."
        )


    # ---------------------------------------------------------------
    # Capture Original CSV Column Order
    # ---------------------------------------------------------------

    original_columns = list(
        dataframe.columns
    )


    # ---------------------------------------------------------------
    # Clean Column Names
    # ---------------------------------------------------------------

    dataframe = clean_column_names(
        dataframe
    )


    # ---------------------------------------------------------------
    # Capture Cleaned Column Order
    # ---------------------------------------------------------------

    CURRENT_COLUMN_ORDER = list(
        dataframe.columns
    )


    # ---------------------------------------------------------------
    # Explicitly Reorder DataFrame
    # ---------------------------------------------------------------

    dataframe = dataframe[
        CURRENT_COLUMN_ORDER
    ]


    # ---------------------------------------------------------------
    # Remove Previous Database
    # ---------------------------------------------------------------

    reset_database()


    # ---------------------------------------------------------------
    # Restore Current Column Order
    #
    # reset_database() clears runtime state, so store it again.
    # ---------------------------------------------------------------

    CURRENT_COLUMN_ORDER = list(
        dataframe.columns
    )


    # ---------------------------------------------------------------
    # Generate Table Name
    # ---------------------------------------------------------------

    table_name = generate_table_name(
        file_path
    )


    # ---------------------------------------------------------------
    # Create SQLite Database
    # ---------------------------------------------------------------

    connection = get_connection()


    try:

        # -----------------------------------------------------------
        # Import DataFrame Into SQLite
        #
        # Pandas keeps the DataFrame column order when creating
        # the SQLite table.
        # -----------------------------------------------------------

        dataframe.to_sql(

            table_name,

            connection,

            if_exists="replace",

            index=False

        )


        connection.commit()


    except Exception as error:

        connection.close()


        raise RuntimeError(

            f"Could not create database table: {error}"

        )


    finally:

        connection.close()


    # ---------------------------------------------------------------
    # Store Current Table
    # ---------------------------------------------------------------

    CURRENT_TABLE_NAME = table_name


    # ---------------------------------------------------------------
    # Build Dynamic Schema
    #
    # IMPORTANT:
    # Iterate through CURRENT_COLUMN_ORDER instead of relying
    # on any other ordering.
    # ---------------------------------------------------------------

    columns = {}


    for column_name in CURRENT_COLUMN_ORDER:

        columns[column_name] = pandas_type_to_sqlite(

            dataframe[column_name].dtype

        )


    # ---------------------------------------------------------------
    # Store Dynamic Schema
    # ---------------------------------------------------------------

    DATABASE_SCHEMA = {

        table_name: {

            "columns": columns,

            "primary_key": None

        }

    }


    # ---------------------------------------------------------------
    # Return Processing Information
    # ---------------------------------------------------------------

    return {

        "table_name": table_name,

        "columns": list(
            CURRENT_COLUMN_ORDER
        ),

        "row_count": len(
            dataframe
        ),

        "schema": DATABASE_SCHEMA

    }


# -------------------------------------------------------------------
# Check Whether Database Is Ready
# -------------------------------------------------------------------

def is_database_ready():
    """
    Check whether a database has been processed.

    Returns:
        bool:
            True if a database is ready.
    """

    return (

        CURRENT_TABLE_NAME is not None

        and

        os.path.exists(
            DATABASE_PATH
        )

    )


# -------------------------------------------------------------------
# Get Current Table
# -------------------------------------------------------------------

def get_current_table():
    """
    Return the currently processed table name.

    Returns:
        str or None:
            Current table name.
    """

    return CURRENT_TABLE_NAME


# -------------------------------------------------------------------
# Get Current Column Order
# -------------------------------------------------------------------

def get_current_column_order():
    """
    Return the current database column order.

    Returns:
        list:
            Column names in the same order as the uploaded CSV.
    """

    return list(
        CURRENT_COLUMN_ORDER
    )


# -------------------------------------------------------------------
# Get Complete Schema
# -------------------------------------------------------------------

def get_schema():
    """
    Return the dynamically extracted database schema.

    Returns:
        dict:
            Current database schema.
    """

    return DATABASE_SCHEMA


# -------------------------------------------------------------------
# Get Tables
# -------------------------------------------------------------------

def get_tables():
    """
    Return the list of currently available tables.

    Returns:
        list:
            Table names.
    """

    return list(
        DATABASE_SCHEMA.keys()
    )


# -------------------------------------------------------------------
# Get Columns
# -------------------------------------------------------------------

def get_columns(
    table_name
):
    """
    Return columns for a table.

    Args:
        table_name (str):
            Table name.

    Returns:
        dict:
            Column names and data types.
    """

    if table_name not in DATABASE_SCHEMA:

        raise ValueError(

            f"Table '{table_name}' does not exist."

        )


    return DATABASE_SCHEMA[
        table_name
    ]["columns"]


# -------------------------------------------------------------------
# Get Schema Text for AI Agent
# -------------------------------------------------------------------

def get_schema_for_prompt():
    """
    Convert the dynamically extracted schema into
    text format for the AI agent.

    Column order is preserved.

    Returns:
        str:
            Schema text.
    """

    if not DATABASE_SCHEMA:

        raise RuntimeError(

            "No database has been processed yet."

        )


    schema_text = []


    for table_name, table_info in DATABASE_SCHEMA.items():

        schema_text.append(

            f"TABLE: {table_name}"

        )


        schema_text.append(
            "COLUMNS:"
        )


        # -----------------------------------------------------------
        # Use Stored Column Order
        # -----------------------------------------------------------

        for column_name in CURRENT_COLUMN_ORDER:

            if column_name in table_info[
                "columns"
            ]:

                data_type = table_info[
                    "columns"
                ][column_name]


                schema_text.append(

                    f"- {column_name} ({data_type})"

                )


        # -----------------------------------------------------------
        # Primary Key
        # -----------------------------------------------------------

        if table_info.get(
            "primary_key"
        ):

            schema_text.append(

                f"PRIMARY KEY: "
                f"{table_info['primary_key']}"

            )


        schema_text.append("")


    return "\n".join(
        schema_text
    )


# -------------------------------------------------------------------
# Validate SQL
# -------------------------------------------------------------------

def validate_sql(
    sql
):
    """
    Perform basic SQL safety validation.

    Only SELECT queries are allowed.

    Args:
        sql (str):
            SQL query.

    Raises:
        ValueError:
            If the query is invalid or unsafe.
    """

    if not sql or not sql.strip():

        raise ValueError(
            "SQL query cannot be empty."
        )


    cleaned_sql = sql.strip()


    # ---------------------------------------------------------------
    # Remove trailing semicolon
    # ---------------------------------------------------------------

    cleaned_sql = cleaned_sql.rstrip(
        ";"
    ).strip()


    # ---------------------------------------------------------------
    # Only SELECT Queries Allowed
    # ---------------------------------------------------------------

    if not cleaned_sql.upper().startswith(
        "SELECT"
    ):

        raise ValueError(

            "Only SELECT queries are allowed."

        )


    # ---------------------------------------------------------------
    # Forbidden SQL Operations
    # ---------------------------------------------------------------

    forbidden_keywords = [

        "INSERT",
        "UPDATE",
        "DELETE",
        "DROP",
        "ALTER",
        "TRUNCATE",
        "CREATE",
        "REPLACE",
        "ATTACH",
        "DETACH"

    ]


    upper_sql = cleaned_sql.upper()


    for keyword in forbidden_keywords:

        if re.search(
            rf"\b{keyword}\b",
            upper_sql
        ):

            raise ValueError(

                f"SQL operation '{keyword}' "
                f"is not allowed."

            )


# -------------------------------------------------------------------
# Execute SQL Query
# -------------------------------------------------------------------

def execute_query(
    sql
):
    """
    Execute a SELECT query against the currently
    processed SQLite database.

    Args:
        sql (str):
            SQL query.

    Returns:
        dict:
            Query result.
    """

    # ---------------------------------------------------------------
    # Check Database
    # ---------------------------------------------------------------

    if not is_database_ready():

        raise RuntimeError(

            "No database has been processed. "
            "Please upload and process a file first."

        )


    # ---------------------------------------------------------------
    # Validate SQL
    # ---------------------------------------------------------------

    validate_sql(
        sql
    )


    connection = None


    try:

        # -----------------------------------------------------------
        # Connect
        # -----------------------------------------------------------

        connection = get_connection()


        cursor = connection.cursor()


        # -----------------------------------------------------------
        # Execute Query
        # -----------------------------------------------------------

        cursor.execute(
            sql
        )


        # -----------------------------------------------------------
        # Get Column Names
        # -----------------------------------------------------------

        columns = [

            description[0]

            for description in cursor.description

        ]


        # -----------------------------------------------------------
        # Get Rows
        # -----------------------------------------------------------

        rows = cursor.fetchall()


        # -----------------------------------------------------------
        # Convert Rows to JSON-Compatible Lists
        # -----------------------------------------------------------

        rows = [

            list(row)

            for row in rows

        ]


        # -----------------------------------------------------------
        # Return Result
        # -----------------------------------------------------------

        return {

            "columns": columns,

            "rows": rows,

            "row_count": len(
                rows
            )

        }


    except sqlite3.Error as error:

        raise RuntimeError(

            f"Database query failed: {error}"

        )


    finally:

        if connection:

            connection.close()


# -------------------------------------------------------------------
# Local Test
# -------------------------------------------------------------------

if __name__ == "__main__":

    print("=" * 60)

    print(
        "             VOICE SQL AGENT DATABASE"
    )

    print("=" * 60)

    print()

    print(
        "Database:"
    )

    print(
        DATABASE_PATH
    )

    print()

    print(
        "No database is created automatically."
    )

    print(
        "Upload and process a CSV file through the application."
    )

    print()

    print(
        "Database Ready:"
    )

    print(
        is_database_ready()
    )

    print()

    print(
        "Current Column Order:"
    )

    print(
        get_current_column_order()
    )