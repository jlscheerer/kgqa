import pandas as pd

from .Database import Database
from .Preferences import Preferences
from .QueryBackend import QueryString
from .QueryGraph import (
    AggregateColumnInfo,
    AnchorEntityColumnInfo,
    QueryGraph,
    HeadVariableColumnInfo,
    PropertyColumnInfo,
    QueryGraphConstantNode,
    QueryGraphEntityConstantNode,
    QueryGraphVariableNode,
)
from .QueryParser import IDConstant, StringConstant
from .SPARQLBackend import SPARQLQuery
from .sparql2sql import sparql2sql


def run_and_rank(query: QueryString, wqg: QueryGraph):
    if isinstance(query, SPARQLQuery):
        if Preferences()["print_sparql"] == "true":
            print("========== [SPARQL] ==========")
            print(query.value)
            print("========== [SPARQL] ==========")

        sql = sparql2sql(query)

        if Preferences()["print_sql"] == "true":
            print("========== [SQL] ==========")
            print(sql.value)
            print("========== [SQL] ==========")

        db = Database()
        results, column_names = db.fetchall(sql.value, return_column_names=True)
    else:
        raise AssertionError(f"cannot run_and_rank query '{query}'")

    user_column_names = [f"{column}" for column in wqg.columns]

    # TODO(jlscheerer) Refactor this code to use dataframes and query only once.
    max_row_score = 0
    annotated_results = []
    for row in results:
        annotated_row = []
        row_score = 1.0
        for id, col in zip(row, wqg.columns):
            # TODO(jlscheerer) Refactor this, because both are the same anyways.
            if isinstance(col, PropertyColumnInfo):
                score = col.node.score(id)
            elif isinstance(col, AnchorEntityColumnInfo):
                node = col.node
                assert isinstance(node.constant, IDConstant) or isinstance(
                    node.constant, StringConstant
                )
                if isinstance(node, QueryGraphEntityConstantNode):
                    score = node.score(id)
                else:
                    score = 1.0
            elif isinstance(col, HeadVariableColumnInfo):
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
                is_entity_id = True
                # TODO(jlscheerer) Check if we actually have an entity_id or a different type.
                if is_entity_id:
                    annotated_row.append(
                        f"{db.get_pid_to_title(id)} ({id}) [f={score}]"
                    )
                else:
                    annotated_row.append(f"{id} [f={score}]")
            if score is not None:
                row_score *= score
        max_row_score = max(max_row_score, row_score)
        annotated_row.append(row_score)
        annotated_results.append(tuple(annotated_row))

    df = pd.DataFrame(annotated_results, columns=user_column_names + ["Score"])
    df["Score"] = df["Score"] / max_row_score
    df = df.sort_values("Score", ascending=False)
    return df, user_column_names + ["Score"]
