from abc import ABC
from typing import Any, Dict, Optional

import pandas as pd
from loguru import logger
from sqlalchemy import MetaData, create_engine, text


class AbstractDatabase(ABC):
    """
    AbstractDatabase is an abstract base class for all database adapters.

    Methods:
        - execute_query(self, query: str) -> Any: Execute a SQL query.
        - connect(self) -> None: Connect to the database.
        - disconnect(self) -> None: Disconnect from the database.
    """

    def __init__(self):
        pass

    def __enter__(self):
        """Enter the context manager, connect to the database."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context manager, disconnect from the database."""
        self.disconnect()

    def execute_query(self, query: str) -> Dict[str, pd.DataFrame]:
        """Execute a SQL query."""
        raise NotImplementedError

    def connect(self) -> None:
        """Connect to the database."""
        pass

    def disconnect(self) -> None:
        """Disconnect from the database."""
        pass


class BaseDatabaseAdapter(AbstractDatabase):
    """
    BaseDatabaseAdapter is a class that implements the AbstractDatabase interface.

    Methods:
        - execute_query(self, query: str) -> Any: Execute a SQL query.
        - connect(self) -> None: Connect to the database.
        - disconnect(self) -> None: Disconnect from the database.
    """

    def __init__(self, kwargs: Dict[str, Any]):
        """
        Initialize the BaseDatabaseAdapter.

        Args:
            kwargs: Dict[str, Any]: The configuration parameters.
        """
        super().__init__()
        self.kwargs = kwargs
        self.schema_info = None
        self.connection_string = kwargs.get("connection_string")
        self.db_type = kwargs.get("db_type", "postgres")
        self.engine = None

    def _get_connection(self) -> Any:
        """
        Get database connection based on database type.

        Returns:
            connection: Any: The database connection object.
        """
        try:
            engine = create_engine(self.connection_string)
            logger.info("SQLAlchemy engine created successfully.")
        except Exception as e:
            logger.error(f"Error creating SQLAlchemy engine: {e}")
            engine = None
        return engine

    def connect(self) -> None:
        """
        Connect to the database.
        """
        if self.engine is None:
            self.engine = self._get_connection()
            if self.engine:
                logger.info(f"Connected to {self.db_type} database")
            else:
                logger.error("Failed to connect to database")

    def disconnect(self) -> None:
        """
        Disconnect from the database.
        """
        if self.engine:
            self.engine.dispose()
            self.engine = None
            logger.info("Disconnected from database")

    def execute_query(
        self,
        sql_statement: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, pd.DataFrame]]:
        """
        Executes a SELECT SQL query and returns the result as a Pandas DataFrame.
        This is the officially supported way.
        """
        if not self.engine:
            logger.error("Engine not available. Cannot execute query.")
            return None

        try:
            # Pandas reads directly from the SQLAlchemy engine
            df = pd.read_sql_query(
                sql=text(sql_statement), con=self.engine, params=params
            )
            return {"data": df}
        except Exception as e:
            logger.error(f"Error executing query to DataFrame: {e}")
            return None

    def get_schema(self) -> Dict[str, Any]:
        """
        Get the schema of the database.
        """
        metadata = MetaData()

        if not self.engine:
            logger.error("Engine not available. Cannot execute query.")
            return None

        try:
            metadata.reflect(bind=self.engine)
            return metadata
        except Exception as e:
            logger.error(f"Error reflecting metadata: {e}")
            return None

    def insert_data(self, table_name: str, data: Dict[str, Any]) -> bool:
        """
        Insert a single row of data into a table.

        Args:
            table_name: Name of the table to insert into
            data: Dictionary of column names and values

        Returns:
            bool: True if successful, False otherwise
        """
        if not self.engine:
            logger.error("Engine not available. Cannot insert data.")
            return False

        # Build INSERT query
        columns = ", ".join(data.keys())
        placeholders = ", ".join([f":{key}" for key in data.keys()])
        query = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})"

        try:
            with self.engine.connect() as conn:
                conn.execute(text(query), data)
                conn.commit()
            return True
        except Exception as e:
            logger.error(f"Error inserting data: {e}")
            return False
