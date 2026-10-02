# app.py

"""
Flask application for the Voice SQL Agent.

Application flow:

1. User uploads a CSV file.
2. User clicks Process.
3. The CSV is converted into a temporary SQLite database.
4. The database schema is extracted dynamically.
5. User enters a natural-language requirement.
6. Groq generates SQL using the extracted schema.
8. SQL is executed against the uploaded data.
9. Results are returned to the frontend.
"""

from flask import (
    Flask,
    request,
    jsonify,
    send_from_directory
)

from agent import generate_sql, generate_result_summary

from database import (
    process_uploaded_file,
    get_schema,
    get_schema_for_prompt,
    get_current_table,
    is_database_ready,
    execute_query
)

import os

# -------------------------------------------------------------------
# Flask Application
# -------------------------------------------------------------------

app = Flask(__name__)


# -------------------------------------------------------------------
# Project Directory
# -------------------------------------------------------------------

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


# -------------------------------------------------------------------
# Upload Directory
# -------------------------------------------------------------------

UPLOAD_FOLDER = os.path.join(
    BASE_DIR,
    "uploads"
)


# -------------------------------------------------------------------
# Create Upload Directory
# -------------------------------------------------------------------

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)


# -------------------------------------------------------------------
# Flask Upload Configuration
# -------------------------------------------------------------------

app.config[
    "UPLOAD_FOLDER"
] = UPLOAD_FOLDER


# -------------------------------------------------------------------
# Home Page
# -------------------------------------------------------------------

@app.route("/")
def home():
    """
    Serve index.html from the project directory.
    """

    return send_from_directory(
        BASE_DIR,
        "index.html"
    )


# -------------------------------------------------------------------
# CSS
# -------------------------------------------------------------------

@app.route("/style.css")
def stylesheet():
    """
    Serve style.css from the project directory.
    """

    return send_from_directory(
        BASE_DIR,
        "style.css"
    )


# -------------------------------------------------------------------
# Process Uploaded File
# -------------------------------------------------------------------

@app.route(
    "/process-file",
    methods=["POST"]
)
def process_file():
    """
    Receive an uploaded CSV file and process it.

    The uploaded file is:

    1. Saved temporarily.
    2. Read by database.py.
    3. Converted into SQLite.
    4. Schema is extracted dynamically.
    5. Processing information is returned.
    """

    try:

        # ------------------------------------------------------------
        # Check File
        # ------------------------------------------------------------

        if "file" not in request.files:

            return jsonify({

                "success": False,

                "error":
                    "Please select a CSV file."

            }), 400


        uploaded_file = request.files[
            "file"
        ]


        # ------------------------------------------------------------
        # Check Filename
        # ------------------------------------------------------------

        if not uploaded_file.filename:

            return jsonify({

                "success": False,

                "error":
                    "No file was selected."

            }), 400


        # ------------------------------------------------------------
        # Validate Extension
        # ------------------------------------------------------------

        filename = uploaded_file.filename

        extension = os.path.splitext(
            filename
        )[1].lower()


        if extension != ".csv":

            return jsonify({

                "success": False,

                "error":
                    "Unsupported file type. "
                    "Please upload a CSV file."

            }), 400


        # ------------------------------------------------------------
        # Remove Previous Uploaded Files
        # ------------------------------------------------------------

        for existing_file in os.listdir(
            UPLOAD_FOLDER
        ):

            existing_path = os.path.join(
                UPLOAD_FOLDER,
                existing_file
            )


            if os.path.isfile(
                existing_path
            ):

                try:

                    os.remove(
                        existing_path
                    )

                except Exception:

                    pass


        # ------------------------------------------------------------
        # Save Uploaded File
        # ------------------------------------------------------------

        file_path = os.path.join(
            UPLOAD_FOLDER,
            filename
        )


        uploaded_file.save(
            file_path
        )


        # ------------------------------------------------------------
        # Process File
        # ------------------------------------------------------------

        result = process_uploaded_file(
            file_path
        )


        # ------------------------------------------------------------
        # Get Schema
        # ------------------------------------------------------------

        schema = get_schema()

        schema_text = get_schema_for_prompt()


        # ------------------------------------------------------------
        # Delete Original CSV After Successful Database Creation
        # ------------------------------------------------------------
        #
        # At this point:
        #   1. The CSV was successfully processed.
        #   2. SQLite database creation completed.
        #   3. The database schema was successfully extracted.
        #
        # The original CSV is no longer required by the application,
        # so remove it from the server's uploads directory.
        #
        # If any earlier processing step fails, execution jumps to the
        # exception handler before reaching this point, so the CSV is
        # retained for error handling/debugging.
        # ------------------------------------------------------------

        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                print(
                    f"Temporary uploaded CSV deleted successfully: "
                    f"{file_path}"
                )
        except OSError as cleanup_error:
            # Database creation was successful, so do not turn a cleanup
            # failure into a failed upload response. Log the issue instead.
            print(
                "Warning: Database was created successfully, but the "
                f"uploaded CSV could not be deleted: {cleanup_error}"
            )

        # ------------------------------------------------------------
        # Return Processing Result
        # ------------------------------------------------------------

        return jsonify({

            "success": True,

            "message":
                "File processed successfully.",

            "filename": filename,

            "table_name":
                result.get(
                    "table_name",
                    ""
                ),

            "columns":
                result.get(
                    "columns",
                    []
                ),

            "row_count":
                result.get(
                    "row_count",
                    0
                ),

            "schema": schema,

            "schema_text": schema_text

        })


    # ----------------------------------------------------------------
    # Error Handling
    # ----------------------------------------------------------------

    except Exception as error:

        return jsonify({

            "success": False,

            "error": str(error)

        }), 500


# -------------------------------------------------------------------
# Database Status
# -------------------------------------------------------------------

@app.route(
    "/database-status",
    methods=["GET"]
)
def database_status():
    """
    Return the current database processing status.
    """

    try:

        ready = is_database_ready()


        if not ready:

            return jsonify({

                "success": True,

                "ready": False,

                "message":
                    "No database has been processed yet."

            })


        return jsonify({

            "success": True,

            "ready": True,

            "table_name":
                get_current_table(),

            "schema":
                get_schema(),

            "schema_text":
                get_schema_for_prompt()

        })


    except Exception as error:

        return jsonify({

            "success": False,

            "error": str(error)

        }), 500


# -------------------------------------------------------------------
# Generate SQL + Execute Query API
# -------------------------------------------------------------------

@app.route(
    "/generate-sql",
    methods=["POST"]
)
def generate_sql_query():
    """
    Receive a natural-language requirement,
    generate SQL using Groq,
    execute the SQL against the uploaded database,
    and return the results.
    """

    try:

        # ------------------------------------------------------------
        # Check Database
        # ------------------------------------------------------------

        if not is_database_ready():

            return jsonify({

                "success": False,

                "error":
                    "Please upload and process a database file first."

            }), 400


        # ------------------------------------------------------------
        # Get JSON Request
        # ------------------------------------------------------------

        data = request.get_json(
            silent=True
        )


        # ------------------------------------------------------------
        # Validate Request
        # ------------------------------------------------------------

        if not data:

            return jsonify({

                "success": False,

                "error":
                    "No data received."

            }), 400


        # ------------------------------------------------------------
        # Get Requirement
        # ------------------------------------------------------------

        requirement = data.get(
            "requirement",
            ""
        )


        # ------------------------------------------------------------
        # Groq is the only AI provider.
        # The provider is fixed here so the frontend cannot switch
        # between Local AI and Groq.
        # ------------------------------------------------------------

        provider = "groq"


        # ------------------------------------------------------------
        # Validate Requirement Type
        # ------------------------------------------------------------

        if not isinstance(
            requirement,
            str
        ):

            return jsonify({

                "success": False,

                "error":
                    "Requirement must be text."

            }), 400


        requirement = requirement.strip()


        # ------------------------------------------------------------
        # Validate Requirement
        # ------------------------------------------------------------

        if not requirement:

            return jsonify({

                "success": False,

                "error":
                    "Please provide a database requirement."

            }), 400


        # ------------------------------------------------------------
        # Generate SQL Using Groq
        # ------------------------------------------------------------

        result = generate_sql(
            requirement,
            provider="groq"
        )


        # ------------------------------------------------------------
        # Get Generated SQL
        # ------------------------------------------------------------

        sql = result.get(
            "sql",
            ""
        )


        if not sql:

            return jsonify({

                "success": False,

                "error":
                    "The AI agent did not generate SQL."

            }), 500


        # ------------------------------------------------------------
        # Execute Generated SQL
        # ------------------------------------------------------------

        query_result = execute_query(
            sql
        )


        # ------------------------------------------------------------
        # Prepare Small Result Sample For AI Summary
        # ------------------------------------------------------------
        #
        # The complete result stays local.
        # At most 10 rows are sent to Groq for the result summary.
        # ------------------------------------------------------------

        result_columns = query_result.get(
            "columns",
            []
        )

        result_rows = query_result.get(
            "rows",
            []
        )

        total_row_count = query_result.get(
            "row_count",
            len(result_rows)
        )

        sample_rows = result_rows[:10]

        sample_records = []

        for row in sample_rows:

            sample_records.append(
                {
                    column: row[index]
                    for index, column in enumerate(result_columns)
                    if index < len(row)
                }
            )


        # ------------------------------------------------------------
        # Generate Result Summary Using Groq
        # ------------------------------------------------------------

        result_summary = ""

        try:

            result_summary = generate_result_summary(
                requirement=requirement,
                sql=sql,
                total_rows=total_row_count,
                sample_rows=sample_records,
                provider="groq"
            )

        except Exception as summary_error:

            # The SQL result must still work if the optional
            # summary AI call fails, for example because of
            # a temporary API/rate-limit problem.
            print(
                "Result summary generation failed:",
                summary_error
            )

            if total_row_count == 0:

                result_summary = (
                    "The query returned no rows."
                )

            elif total_row_count == 1:

                result_summary = (
                    "The query returned 1 row."
                )

            else:

                result_summary = (
                    f"The query returned {total_row_count} rows."
                )


        # ------------------------------------------------------------
        # Return Complete Response
        # ------------------------------------------------------------

        return jsonify({

            "success": True,

            "requirement":
                requirement,

            "provider":
                provider,

            "table_name":
                get_current_table(),

            "sql":
                sql,

            "explanation":
                result.get(
                    "explanation",
                    ""
                ),

            "tables_used":
                result.get(
                    "tables_used",
                    []
                ),

            "columns_used":
                result.get(
                    "columns_used",
                    []
                ),

            "operations":
                result.get(
                    "operations",
                    []
                ),

            "result_summary":
                result_summary,

            "result": {

                "columns":
                    query_result.get(
                        "columns",
                        []
                    ),

                "rows":
                    query_result.get(
                        "rows",
                        []
                    ),

                "row_count":
                    query_result.get(
                        "row_count",
                        0
                    )

            }

        })


    # ----------------------------------------------------------------
    # Error Handling
    # ----------------------------------------------------------------

    except Exception as error:

        return jsonify({

            "success": False,

            "error": str(error)

        }), 500


# -------------------------------------------------------------------
# Health Check
# -------------------------------------------------------------------

@app.route(
    "/health",
    methods=["GET"]
)
def health_check():
    """
    Check whether the Flask server is running.
    """

    return jsonify({

        "status":
            "running",

        "service":
            "Voice SQL Agent",

        "database_ready":
            is_database_ready()

    })


# -------------------------------------------------------------------
# Start Flask Server
# -------------------------------------------------------------------

if __name__ == "__main__":

    print("=" * 60)

    print(
        "                 VOICE SQL AGENT"
    )

    print("=" * 60)

    print()

    print(
        "Server starting..."
    )

    print()

    print(
        "URL: http://127.0.0.1:5000"
    )

    print()

    print(
        "Health: http://127.0.0.1:5000/health"
    )

    print()

    print(
        "Database: Dynamic SQLite"
    )

    print()

    print(
        "AI Provider: Groq API only"
    )

    print()

    print(
        "Speech-to-Text: Browser Speech Recognition"
    )

    print()

    print(
        "Groq Model: openai/gpt-oss-20b"
    )

    print()

    print(
        "Supported Upload: CSV"
    )

    print()

    print("=" * 60)


    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )
