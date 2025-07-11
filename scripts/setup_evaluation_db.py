#!/usr/bin/env python3
"""
Script to set up the evaluation database.
Creates a separate database for evaluation results while using the same PostgreSQL instance.
"""

import os
import sys
from pathlib import Path

import psycopg2
from psycopg2 import sql
from sqlalchemy import create_engine, text
from loguru import logger

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))


def create_evaluation_database():
    """Create the evaluation database if it doesn't exist."""

    # Connection parameters for PostgreSQL
    pg_host = os.getenv("PG_HOST", "localhost")
    pg_port = os.getenv("PG_PORT", "5432")
    pg_user = os.getenv("PG_USER", "postgres")
    pg_password = os.getenv("PG_PASSWORD", "example")
    pg_eval_db = os.getenv("PG_EVAL_DB", "evaluation")

    # Connect to PostgreSQL server (default postgres database)
    conn = None
    cursor = None

    try:
        # Connect to default postgres database to create the evaluation database
        conn = psycopg2.connect(
            host=pg_host,
            port=pg_port,
            user=pg_user,
            password=pg_password,
            database="postgres",
        )
        conn.autocommit = True
        cursor = conn.cursor()

        # Check if database exists
        cursor.execute("SELECT 1 FROM pg_database WHERE datname = %s", (pg_eval_db,))
        exists = cursor.fetchone()

        if not exists:
            # Create database
            cursor.execute(
                sql.SQL("CREATE DATABASE {}").format(sql.Identifier(pg_eval_db))
            )
            logger.info(f"Created database '{pg_eval_db}'")
        else:
            logger.info(f"Database '{pg_eval_db}' already exists")

    except Exception as e:
        logger.error(f"Error creating database: {e}")
        raise
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


def migrate_metadata_columns(engine):
    """Rename metadata columns to meta_data if they exist."""

    migrations = [
        ("evaluation_runs", "metadata", "meta_data"),
        ("test_results", "metadata", "meta_data"),
    ]

    with engine.begin() as conn:
        for table_name, old_col, new_col in migrations:
            # Check if old column exists
            result = conn.execute(
                text("""
                SELECT column_name
                FROM information_schema.columns
                WHERE table_name = :table_name AND column_name = :old_col
            """),
                {"table_name": table_name, "old_col": old_col},
            )

            if result.fetchone():
                # Rename column
                conn.execute(
                    text(
                        f"ALTER TABLE {table_name} RENAME COLUMN {old_col} TO {new_col}"
                    )
                )
                logger.info(f"Renamed {table_name}.{old_col} to {new_col}")


def create_evaluation_tables():
    """Create evaluation tables using SQLAlchemy."""

    # Connection parameters
    pg_host = os.getenv("PG_HOST", "localhost")
    pg_port = os.getenv("PG_PORT", "5432")
    pg_user = os.getenv("PG_USER", "postgres")
    pg_password = os.getenv("PG_PASSWORD", "example")
    pg_eval_db = os.getenv("PG_EVAL_DB", "evaluation")

    # Create connection string
    connection_string = (
        f"postgresql://{pg_user}:{pg_password}@{pg_host}:{pg_port}/{pg_eval_db}"
    )

    # Create engine
    engine = create_engine(connection_string)

    # First, try to migrate existing columns if needed
    try:
        migrate_metadata_columns(engine)
    except Exception as e:
        logger.warning(
            f"Migration check failed (this is OK for new installations): {e}"
        )

    # SQL for creating tables
    create_tables_sql = """
    -- Create UUID extension if not exists
    CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

    -- Main evaluation runs table
    CREATE TABLE IF NOT EXISTS evaluation_runs (
        id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
        run_type VARCHAR(50) NOT NULL,
        evaluation_category VARCHAR(50),
        stage VARCHAR(50),
        git_commit VARCHAR(40),
        branch VARCHAR(100),
        started_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
        completed_at TIMESTAMP,
        total_tests INTEGER,
        passed_tests INTEGER,
        failed_tests INTEGER,
        model_name VARCHAR(100),
        model_temperature FLOAT,
        prompt_version VARCHAR(50),
        fixtures_used TEXT[],
        meta_data JSONB,
        created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
    );

    -- Individual test results
    CREATE TABLE IF NOT EXISTS test_results (
        id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
        run_id UUID NOT NULL REFERENCES evaluation_runs(id) ON DELETE CASCADE,
        test_name VARCHAR(200) NOT NULL,
        test_type VARCHAR(50),
        question TEXT,
        expected_response TEXT,
        actual_response TEXT,
        passed BOOLEAN NOT NULL,
        execution_time_ms INTEGER,
        judge_scores JSONB,
        judge_reasoning TEXT,
        error_message TEXT,
        meta_data JSONB,
        created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
    );

    -- Aggregated metrics per run
    CREATE TABLE IF NOT EXISTS evaluation_metrics (
        id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
        run_id UUID NOT NULL REFERENCES evaluation_runs(id) ON DELETE CASCADE,
        metric_type VARCHAR(50) NOT NULL,
        average_score FLOAT,
        min_score FLOAT,
        max_score FLOAT,
        std_deviation FLOAT,
        percentile_25 FLOAT,
        percentile_50 FLOAT,
        percentile_75 FLOAT,
        created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
    );

    -- Tool agent specific results
    CREATE TABLE IF NOT EXISTS tool_agent_results (
        id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
        test_result_id UUID NOT NULL REFERENCES test_results(id) ON DELETE CASCADE,
        tools_used TEXT[],
        tool_outputs JSONB,
        execution_delay_ms INTEGER,
        created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
    );

    -- SQL test specific results
    CREATE TABLE IF NOT EXISTS sql_test_results (
        id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
        test_result_id UUID NOT NULL REFERENCES test_results(id) ON DELETE CASCADE,
        stage VARCHAR(50),
        schema_context JSONB,
        sql_query TEXT,
        query_plan TEXT,
        created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
    );

    -- Create indexes for better query performance
    CREATE INDEX IF NOT EXISTS idx_evaluation_runs_run_type ON evaluation_runs(run_type);
    CREATE INDEX IF NOT EXISTS idx_evaluation_runs_created_at ON evaluation_runs(created_at);
    CREATE INDEX IF NOT EXISTS idx_evaluation_runs_git_commit ON evaluation_runs(git_commit);
    CREATE INDEX IF NOT EXISTS idx_test_results_run_id ON test_results(run_id);
    CREATE INDEX IF NOT EXISTS idx_test_results_passed ON test_results(passed);
    CREATE INDEX IF NOT EXISTS idx_test_results_test_name ON test_results(test_name);
    CREATE INDEX IF NOT EXISTS idx_evaluation_metrics_run_id ON evaluation_metrics(run_id);
    CREATE INDEX IF NOT EXISTS idx_evaluation_metrics_metric_type ON evaluation_metrics(metric_type);

    -- Create updated_at trigger function
    CREATE OR REPLACE FUNCTION update_updated_at_column()
    RETURNS TRIGGER AS $$
    BEGIN
        NEW.updated_at = CURRENT_TIMESTAMP;
        RETURN NEW;
    END;
    $$ language 'plpgsql';

    -- Create trigger for evaluation_runs
    DROP TRIGGER IF EXISTS update_evaluation_runs_updated_at ON evaluation_runs;
    CREATE TRIGGER update_evaluation_runs_updated_at
        BEFORE UPDATE ON evaluation_runs
        FOR EACH ROW
        EXECUTE FUNCTION update_updated_at_column();
    """

    try:
        with engine.begin() as conn:
            # Execute the entire SQL script as one block
            # This preserves the trigger function with its $$ delimiters
            conn.execute(text(create_tables_sql))

        logger.info("Successfully created all evaluation tables")

    except Exception as e:
        logger.error(f"Error creating tables: {e}")
        raise
    finally:
        engine.dispose()


def verify_setup():
    """Verify the database setup by checking tables exist."""

    pg_host = os.getenv("PG_HOST", "localhost")
    pg_port = os.getenv("PG_PORT", "5432")
    pg_user = os.getenv("PG_USER", "postgres")
    pg_password = os.getenv("PG_PASSWORD", "example")
    pg_eval_db = os.getenv("PG_EVAL_DB", "evaluation")

    connection_string = (
        f"postgresql://{pg_user}:{pg_password}@{pg_host}:{pg_port}/{pg_eval_db}"
    )
    engine = create_engine(connection_string)

    try:
        with engine.connect() as conn:
            # Check tables exist
            result = conn.execute(
                text("""
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'public'
                AND table_type = 'BASE TABLE'
                ORDER BY table_name;
            """)
            )

            tables = [row[0] for row in result]

            expected_tables = [
                "evaluation_runs",
                "test_results",
                "evaluation_metrics",
                "tool_agent_results",
                "sql_test_results",
            ]

            logger.info("Found tables:")
            for table in tables:
                logger.info(f"  - {table}")

            missing_tables = set(expected_tables) - set(tables)
            if missing_tables:
                logger.error(f"Missing tables: {missing_tables}")
                return False

            logger.info("All expected tables exist!")
            return True

    except Exception as e:
        logger.error(f"Error verifying setup: {e}")
        return False
    finally:
        engine.dispose()


def main():
    """Main function to set up the evaluation database."""

    logger.info("Starting evaluation database setup...")

    # Load environment variables
    from dotenv import load_dotenv

    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        logger.info(f"Loaded environment from {env_path}")

    try:
        # Step 1: Create database
        create_evaluation_database()

        # Step 2: Create tables
        create_evaluation_tables()

        # Step 3: Verify setup
        if verify_setup():
            logger.success("Evaluation database setup completed successfully!")
        else:
            logger.error("Database setup verification failed")
            sys.exit(1)

    except Exception as e:
        logger.error(f"Setup failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
