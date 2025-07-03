from unittest.mock import patch

import pandas as pd
import pytest

from src.agent.adapters.database import BaseDatabaseAdapter


@pytest.fixture
def database_instance():
    kwargs = {
        "connection_string": "postgresql://test:test@localhost:5432/test",
        "db_type": "postgres",
    }
    return BaseDatabaseAdapter(kwargs)


class TestDatabase:
    def test_database_init(self, database_instance):
        assert (
            database_instance.connection_string
            == "postgresql://test:test@localhost:5432/test"
        )
        assert database_instance.db_type == "postgres"

    @patch("src.agent.adapters.database.pd.read_sql_query")
    def test_execute_query(self, mock_read_sql_query, database_instance):
        mock_read_sql_query.return_value = pd.DataFrame()

        with database_instance as db:
            result = db.execute_query("SELECT * FROM your_table")

        pd.testing.assert_frame_equal(result["data"], pd.DataFrame())

    @patch(
        "src.agent.adapters.database.create_engine",
        side_effect=Exception("Database connection failed"),
    )
    def test_connection_error(self, mock_create_engine, database_instance):
        with database_instance as db:
            result = db.execute_query("SELECT * FROM your_table")
            assert result is None

    def test_execute_query_no_engine(self, database_instance):
        result = database_instance.execute_query("SELECT * FROM your_table")
        assert result is None
