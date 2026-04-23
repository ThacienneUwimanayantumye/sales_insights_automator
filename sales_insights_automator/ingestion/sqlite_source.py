"""
SQLite data source connector.

Executes a SQL query against a local SQLite database and returns the result
as a DataFrame.  Accepts either a raw SQL string or just a table name —
if a table name is given the connector issues a ``SELECT *`` automatically.
"""

import sqlite3
import os
from typing import Optional

import pandas as pd

from ingestion.base import DataSource, DataSourceError


class SQLiteSource(DataSource):
    """Loads data from a SQLite database via a SQL query.

    Parameters
    ----------
    db_path : str
        Path to the ``.db`` / ``.sqlite`` file.
    query : str, optional
        A full SQL query to execute.  Mutually exclusive with ``table``.
    table : str, optional
        Table name.  If provided (and ``query`` is not), the connector
        executes ``SELECT * FROM <table>``.
    params : tuple or list, optional
        Positional parameters for parameterised queries, e.g.
        ``("2024-01-01",)`` for ``WHERE date >= ?``.

    Notes
    -----
    Exactly one of ``query`` or ``table`` must be provided.

    Examples
    --------
    >>> source = SQLiteSource("data/samples/sales.db", table="sales")
    >>> df = source.load_validated()

    >>> source = SQLiteSource(
    ...     "data/samples/sales.db",
    ...     query="SELECT * FROM sales WHERE region = 'West'",
    ... )
    >>> df = source.load_validated()
    """

    def __init__(
        self,
        db_path: str,
        query: Optional[str] = None,
        table: Optional[str] = None,
        params: Optional[tuple] = None,
    ) -> None:
        if query is None and table is None:
            raise ValueError("Provide either 'query' or 'table', not neither.")
        if query is not None and table is not None:
            raise ValueError("Provide either 'query' or 'table', not both.")

        self.db_path = db_path
        self.query = query if query else f"SELECT * FROM {table}"
        self.params = params or ()

    # ------------------------------------------------------------------ #

    def validate(self) -> bool:
        """Return True if the database file exists and is readable."""
        if not os.path.isfile(self.db_path):
            print(f"[SQLiteSource] Database file not found: {self.db_path}")
            return False
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute("SELECT 1")
            conn.close()
        except sqlite3.Error as exc:
            print(f"[SQLiteSource] Cannot open database: {exc}")
            return False
        return True

    def load(self) -> pd.DataFrame:
        """Execute the SQL query and return the result as a DataFrame.

        Returns
        -------
        pd.DataFrame

        Raises
        ------
        DataSourceError
            If the database cannot be opened or the query fails.
        """
        try:
            conn = sqlite3.connect(self.db_path)
            df = pd.read_sql_query(self.query, conn, params=self.params)
            conn.close()
        except sqlite3.Error as exc:
            raise DataSourceError(
                f"SQLite error on '{self.db_path}': {exc}"
            ) from exc
        except Exception as exc:
            raise DataSourceError(
                f"Failed to load from '{self.db_path}': {exc}"
            ) from exc

        print(
            f"[SQLiteSource] Loaded {len(df):,} rows from '{self.db_path}' "
            f"using query: {self.query[:80]}{'...' if len(self.query) > 80 else ''}"
        )
        return df

    def list_tables(self) -> list[str]:
        """Return the names of all tables in the database.

        Useful for exploration before deciding which table / query to use.
        """
        if not self.validate():
            raise DataSourceError(f"Cannot connect to '{self.db_path}'")
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;"
        )
        tables = [row[0] for row in cursor.fetchall()]
        conn.close()
        return tables

    def describe(self) -> str:
        return f"SQLiteSource(db='{self.db_path}', query='{self.query[:60]}...')"
