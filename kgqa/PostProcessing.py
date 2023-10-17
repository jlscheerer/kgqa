import pandas as pd

from kgqa.Database import Database
from kgqa.QueryGraph import ExecutableQueryGraph, QueryStatistics


def run_and_rank(sql_query: str, wqg: ExecutableQueryGraph, stats: QueryStatistics):
    db = Database()
    results, column_names = db.fetchall(sql_query, return_column_names=True)

    if column_names != [f"{column}" for column in stats.columns]:
        raise AssertionError("received unexpected table format from database")

    df = pd.DataFrame(results, columns=column_names)
    return results, column_names
