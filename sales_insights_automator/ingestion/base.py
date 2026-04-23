"""
Base class for all data source connectors.

Every connector in the ingestion layer must inherit from DataSource and
implement the two abstract methods defined here.  This enforces a consistent
interface across all connectors, making the pipeline trivially extensible —
adding a new data source means writing one new class, nothing else changes.
"""

from abc import ABC, abstractmethod

import pandas as pd


class DataSource(ABC):
    """Abstract base class for all ingestion connectors.

    Subclasses represent a single *type* of data source (CSV file, SQLite
    database, cloud API, etc.).  The rest of the pipeline only interacts with
    the interface defined here, so sources are interchangeable.

    Typical usage
    -------------
    source = CSVSource(filepath="data/samples/sales.csv")
    df = source.load()
    """

    # ------------------------------------------------------------------ #
    # Abstract interface — every connector must implement these           #
    # ------------------------------------------------------------------ #

    @abstractmethod
    def load(self) -> pd.DataFrame:
        """Load data from the source and return it as a DataFrame.

        Returns
        -------
        pd.DataFrame
            Raw, unprocessed data exactly as it comes from the source.

        Raises
        ------
        DataSourceError
            If the data cannot be loaded (file missing, bad query, auth
            failure, etc.).
        """

    @abstractmethod
    def validate(self) -> bool:
        """Check whether the source is reachable / readable before loading.

        Returns
        -------
        bool
            True if the source can be accessed, False otherwise.
        """

    # ------------------------------------------------------------------ #
    # Concrete helpers available to all subclasses                        #
    # ------------------------------------------------------------------ #

    def describe(self) -> str:
        """Return a human-readable description of this connector.

        Subclasses may override this to provide richer information, but the
        default implementation is sufficient for most connectors.
        """
        return f"{self.__class__.__name__} connector"

    def load_validated(self) -> pd.DataFrame:
        """Validate then load in one call.

        This is the recommended entry point for pipeline code that wants a
        single, safe method to call rather than having to remember to call
        ``validate()`` separately.

        Raises
        ------
        DataSourceError
            If ``validate()`` returns False.
        """
        if not self.validate():
            raise DataSourceError(
                f"{self.describe()} failed validation. "
                "Check that the source exists and is accessible."
            )
        return self.load()


class DataSourceError(Exception):
    """Raised when a data source cannot be loaded or validated."""
