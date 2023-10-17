from kgqa.Database import Database
from kgqa.QueryGraph import ExecutableQueryGraph, QueryStatistics


def run_and_rank(sql_query: str, wqg: ExecutableQueryGraph, stats: QueryStatistics):
    db = Database()
    results, column_names = db.fetchall(sql_query, return_column_names=True)
    return results, column_names
