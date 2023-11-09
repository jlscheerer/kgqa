from typing import Dict
import pandas as pd

from kgqa.Database import Database
from kgqa.Preferences import Preferences
from kgqa.QueryBackend import QueryString
from kgqa.QueryGraph import (
    AggregateColumnInfo,
    AnchorEntityColumnInfo,
    ColumnInfo,
    ExecutableQueryGraph,
    HeadEntityColumnInfo,
    PropertyColumnInfo,
    QueryStatistics,
)
from kgqa.QueryParser import IDConstant, StringConstant
from kgqa.SPARQLBackend import SPARQLQuery
from kgqa.SQLBackend import SQLQuery
from kgqa.sparql2sql import sparql2sql


def run_and_rank(query: QueryString, wqg: ExecutableQueryGraph, stats: QueryStatistics):
    if isinstance(query, SQLQuery):
        db = Database()
        results, column_names = db.fetchall(query.value, return_column_names=True)

        if column_names != [f"{column}" for column in stats.columns]:
            raise AssertionError("received unexpected table format from database")
    elif isinstance(query, SPARQLQuery):
        col2name: Dict[ColumnInfo, str] = stats.meta["col2name"]

        if Preferences()["print_sparql"] == "true":
            print("========== [SPARQL] ==========")
            print(query.value)
            print("========== [SPARQL] ==========")

        sql = sparql2sql(query)
        db = Database()
        results, column_names = db.fetchall(sql.value, return_column_names=True)

        if column_names != [
            f"?Z{column.aggregate_index}"
            if isinstance(column, AggregateColumnInfo)
            else col2name[column]
            for column in stats.columns
        ]:
            raise AssertionError("received unexpected table format from database")
    else:
        raise AssertionError(f"cannot run_and_rank query '{query}'")

    user_column_names = [f"{column}" for column in stats.columns]

    # TODO(jlscheerer) Refactor this code to use dataframes and query only once.
    annotated_results = []
    for row in results:
        annotated_row = []
        row_score = 1.0
        for id, col in zip(row, stats.columns):
            # TODO(jlscheerer) Refactor this, because both are the same anyways.
            if isinstance(col, PropertyColumnInfo):
                score = stats.pid2scores[col.predicate.query_name()][id]
            elif isinstance(col, AnchorEntityColumnInfo):
                node = wqg.nodes[col.index.value]
                assert isinstance(node.value, IDConstant) or isinstance(
                    node.value, StringConstant
                )
                score = stats.qid2scores[node.value.value][id]
            elif isinstance(col, HeadEntityColumnInfo):
                score = None  # Entity is retrieved. Thus, we have no score
            elif isinstance(col, AggregateColumnInfo):
                score = None
            else:
                assert False
            if isinstance(col, AggregateColumnInfo):
                annotated_row.append(
                    id
                )  # id is actually the value of the aggregate in this case.
            else:
                annotated_row.append(f"{db.get_pid_to_title(id)} ({id}) [f={score}]")
            if score is not None:
                row_score *= score
        annotated_row.append(row_score)
        annotated_results.append(tuple(annotated_row))

    df = pd.DataFrame(annotated_results, columns=user_column_names + ["Score"])
    df = df.sort_values("Score", ascending=False)
    return df, user_column_names + ["Score"]
