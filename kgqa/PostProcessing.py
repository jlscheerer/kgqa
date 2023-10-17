import pandas as pd

from kgqa.Database import Database
from kgqa.QueryBackend import QueryString
from kgqa.QueryGraph import ExecutableQueryGraph, QueryStatistics
from kgqa.SQLBackend import SQLQuery


def run_and_rank(query: QueryString, wqg: ExecutableQueryGraph, stats: QueryStatistics):
    assert isinstance(query, SQLQuery)

    db = Database()
    results, column_names = db.fetchall(query.value, return_column_names=True)

    if column_names != [f"{column}" for column in stats.columns]:
        raise AssertionError("received unexpected table format from database")

    df = pd.DataFrame(results, columns=column_names)
    return results, column_names
