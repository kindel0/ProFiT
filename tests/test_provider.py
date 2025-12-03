"""
Tests for the data provider implementations.
"""
import os
import pandas as pd
import pytest

from profit.data.provider import CSVProvider, ParquetProvider

# Define paths to test data
BASE_DIR = os.path.dirname(__file__)
VALID_CSV = os.path.join(BASE_DIR, 'data', 'sample_data.csv')
VALID_PARQUET = os.path.join(BASE_DIR, 'data', 'sample_data.parquet')
BAD_COLS_CSV = os.path.join(BASE_DIR, 'data', 'bad_columns.csv')
BAD_INDEX_CSV = os.path.join(BASE_DIR, 'data', 'bad_index_no_date.csv')
NON_EXISTENT_FILE = 'non_existent.csv'


@pytest.mark.parametrize("provider_class, path", [
    (CSVProvider, VALID_CSV),
    (ParquetProvider, VALID_PARQUET)
])
def test_valid_data_loading(provider_class, path):
    """
    Tests that both CSV and Parquet providers can load a valid data file.
    """
    provider = provider_class(path=path)
    df = provider.load()

    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert isinstance(df.index, pd.DatetimeIndex)
    assert 'close' in df.columns # Check that columns were lowercased


@pytest.mark.parametrize("provider_class, path", [
    (CSVProvider, BAD_COLS_CSV),
    (ParquetProvider, VALID_PARQUET) # Parquet test requires creating a bad parquet file
])
def test_missing_columns_raises_error(provider_class, path, request):
    """
    Tests that a ValueError is raised if the data is missing required columns.
    """
    if provider_class == ParquetProvider:
        # Create a parquet file with bad columns for the test
        df = pd.read_parquet(path)
        df.columns = [col.lower() for col in df.columns]
        df_bad = df.drop(columns=['high'])
        bad_path = os.path.join(BASE_DIR, 'data', 'bad_columns.parquet')
        df_bad.to_parquet(bad_path)
        path = bad_path # Point test to the new bad file
        
        def cleanup():
            os.remove(bad_path)
        request.addfinalizer(cleanup)

    provider = provider_class(path=path)
    with pytest.raises(ValueError, match="missing required columns"):
        provider.load()


def test_csv_invalid_index_raises_error():
    """
    Tests that a ValueError is raised if the data does not have a DatetimeIndex.
    """
    provider = CSVProvider(path=BAD_INDEX_CSV)
    # The validation happens after read_csv which doesn't create a DatetimeIndex here
    with pytest.raises(ValueError, match="must have a DatetimeIndex"):
        provider.load()


@pytest.mark.parametrize("provider_class", [CSVProvider, ParquetProvider])
def test_non_existent_file_raises_error(provider_class):
    """
    Tests that a FileNotFoundError is raised for a non-existent file path.
    """
    provider = provider_class(path=NON_EXISTENT_FILE)
    with pytest.raises(FileNotFoundError):
        provider.load()
