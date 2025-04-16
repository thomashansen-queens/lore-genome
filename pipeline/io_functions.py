"""
This module handles reading/writing results from the API.
"""
import logging
from pathlib import Path
from typing import Callable
import pandas as pd

def load_or_fetch_df(
    *args,
    cache_path: Path = '.',
    fetch_func: Callable = None,
    allow_fetch: bool = True,
    **kwargs,
) -> pd.DataFrame:
    """
    Checks if a Pickle or CSV cache file exists and loads it. Otherwise,
    calls the fetch function to generate the DataFrame, then saves it
    to both Pickle and CSV for future use.

    :param cache_path (Path): Path to the cache file. Automatic file extension.
    :param fetch_func (Callable): Function to call to fetch the data if missing.
    :param allow_fetch (bool): Whether to allow fetching if no cache is found.
    :param *args, **kwargs: Arguments passed to the fetch_func.
    :return pd.DataFrame: The loaded or freshly fetched DataFrame.
    """
    pkl_path = cache_path.with_suffix('.pkl')
    csv_path = cache_path.with_suffix('.csv')
    if pkl_path.exists():
        logging.info("Pickle cache found at %s. Loading...", pkl_path)
        return pd.read_pickle(pkl_path)

    if csv_path.exists():
        logging.info("CSV cache found at %s. Loading...", csv_path)
        logging.warning("Loading CSV does not preserve dtypes!")
        return pd.read_csv(csv_path)

    if not allow_fetch:
        logging.error("Could not find %s and fetching is disabled.", cache_path)
        raise FileNotFoundError("Cached data not found. Set allow_fetch=True to fetch data.")

    if fetch_func is not None:
        logging.warning("No cached data found at %s. Fetching data...", cache_path)
        df = fetch_func(*args, **kwargs)
        # Save to both Parquet and CSV
        df.to_pickle(pkl_path)
        df.to_csv(csv_path, index=False)
        logging.info("Data cached to %s and %s.", pkl_path, csv_path)
        return df

    logging.error("Could not find %s.pkl or .csv and no fetch_func provided.", cache_path)
    return pd.DataFrame()


def load_or_fetch_text(
    *args,
    cache_path: Path = '.',
    fetch_func: Callable = None,
    allow_fetch: bool = True,
    **kwargs,
) -> str:
    """
    Checks if a text file exists and loads it. Otherwise, calls the fetch
    function to generate the data, then saves it to the cache for future use.

    :param cache_path (Path): Path to the cache file.
    :param fetch_func (Callable): Function to call to fetch the data if missing.
    :param allow_fetch (bool): Whether to allow fetching if no cache is found.
    :param *args, **kwargs: Arguments passed to the fetch_func.
    :return str: The loaded or freshly fetched data.
    """
    if cache_path.exists():
        logging.info("Cache found at %s. Loading...", cache_path)
        return cache_path.read_text()

    if not allow_fetch:
        logging.error("Could not find %s and fetching is disabled.", cache_path)
        raise FileNotFoundError("Cached data not found. Set allow_fetch=True to fetch data.")

    if fetch_func is not None:
        logging.warning("No cached data found at %s. Fetching data...", cache_path)
        data = fetch_func(*args, **kwargs)
        # Save to cache
        cache_path.write_text(data)
        logging.info("Data cached to %s", cache_path)
        return data

    logging.error("Could not find %s and no fetch_func provided.", cache_path)
    return ""
