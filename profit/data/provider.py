"""
Data provider interfaces and implementations.

This module defines the abstract interface for data providers and provides
concrete implementations for loading financial time-series data from various
file formats like CSV and Parquet.
"""
from abc import ABC, abstractmethod
from typing import List

import pandas as pd


class DataProvider(ABC):
    """
    Abstract base class for all data providers.

    It defines a common interface for loading data and enforces basic validation
    checks to ensure the data is in the expected format.
    """
    REQUIRED_COLUMNS: List[str] = ["open", "high", "low", "close", "volume"]

    def __init__(self, path: str):
        """
        Initializes the data provider.

        Args:
            path (str): The path to the data source file.
        """
        self._path = path

    @abstractmethod
    def load(self) -> pd.DataFrame:
        """
        Loads the data from the source, performs validation, and returns it.

        This method must be implemented by all concrete subclasses.

        Returns:
            pd.DataFrame: A validated DataFrame containing the time-series data.
        """
        raise NotImplementedError

    def _validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Performs validation on the loaded DataFrame.

        - Converts all column names to lowercase.
        - Checks for the presence of required columns (OHLCV).
        - Ensures the DataFrame has a DatetimeIndex.

        Args:
            df (pd.DataFrame): The DataFrame to validate.

        Returns:
            pd.DataFrame: The validated DataFrame.

        Raises:
            ValueError: If validation fails (e.g., missing columns or wrong
                index type).
        """
        df.columns = [col.lower() for col in df.columns]

        if not all(col in df.columns for col in self.REQUIRED_COLUMNS):
            missing = set(self.REQUIRED_COLUMNS) - set(df.columns)
            raise ValueError(f"DataFrame is missing required columns: {missing}")

        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have a DatetimeIndex.")

        return df


class ParquetProvider(DataProvider):
    """
    A data provider for loading time-series data from a Parquet file.
    """

    def load(self) -> pd.DataFrame:
        """
        Loads data from the specified Parquet file.

        Returns:
            pd.DataFrame: A validated DataFrame with the loaded data.

        Raises:
            FileNotFoundError: If the file at `self._path` does not exist.
        """
        df = pd.read_parquet(self._path)
        return self._validate(df)


class CSVProvider(DataProvider):
    """
    A data provider for loading time-series data from a CSV file.

    It assumes that the first column of the CSV is the timestamp index.
    """

    def load(self) -> pd.DataFrame:
        """
        Loads data from the specified CSV file.

        Returns:
            pd.DataFrame: A validated DataFrame with the loaded data.

        Raises:
            FileNotFoundError: If the file at `self._path` does not exist.
        """
        df = pd.read_csv(self._path, index_col=0, parse_dates=True, date_format="%Y-%m-%d")
        return self._validate(df)
