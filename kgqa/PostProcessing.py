from typing import Dict
import pandas as pd

from kgqa.Database import Database
from kgqa.QueryBackend import QueryString
from kgqa.QueryGraph import ColumnInfo, ExecutableQueryGraph, QueryStatistics
from kgqa.SPARQLBackend import SPARQLQuery
from kgqa.SQLBackend import SQLQuery
from kgqa.sparql2sql import sparql2sql


def run_and_rank(query: QueryString, wqg: ExecutableQueryGraph, stats: QueryStatistics):
    if isinstance(query, SQLQuery):
        db = Database()
        results, column_names = db.fetchall(query.value, return_column_names=True)

        if column_names != [f"{column}" for column in stats.columns]:
            raise AssertionError("received unexpected table format from database")

        df = pd.DataFrame(results, columns=column_names)
        return results, column_names
    elif isinstance(query, SPARQLQuery):
        col2name: Dict[ColumnInfo, str] = stats.meta["col2name"]

        print("========== [SPARQL] ==========")
        print(query.value)
        print("========== [SPARQL] ==========")

        sql = sparql2sql(query)
        db = Database()
        results, column_names = db.fetchall(sql.value, return_column_names=True)

        if column_names != [col2name[column] for column in stats.columns]:
            raise AssertionError("received unexpected table format from database")

        df = pd.DataFrame(results, columns=column_names)
        return results, column_names
    else:
        raise AssertionError(f"cannot run_and_rank query '{query}'")
