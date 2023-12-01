import inspect
from dataclasses import dataclass
from typing import List
from typing_extensions import override

from .QueryBackend import QueryBackend, QueryString
from .QueryGraph import (
    QueryGraphPropertyNode,
    QueryGraphEntityConstantNode,
    AggregateColumnInfo,
    AnchorEntityColumnInfo,
    ColumnInfo,
    HeadVariableColumnInfo,
    PropertyColumnInfo,
    QueryGraph,
)


@dataclass
class SPARQLQuery(QueryString):
    pass


class SPARQLBackend(QueryBackend):
    @override
    def to_query(self, emit_labels: bool = False) -> SPARQLQuery:
        if emit_labels:
            raise AssertionError("unsupported option 'emit_labels' for SPARQLBackend")

        SELECT = self._construct_select()
        WHERE = self._construct_where()
        FILTER = self._construct_filter()
        GROUP_BY = str()
        if self.requires_aggregation():
            GROUP_BY = f"GROUP BY {self._construct_group_by()}"
        query = self._dedent_query(
            f"""SELECT {SELECT}
                WHERE
                {{
                    {WHERE}
                    FILTER(\t{FILTER})
                    SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
                }}
                {GROUP_BY}"""
        )
        return SPARQLQuery(value=query, col2name=self.col2name)

    def _construct_select(self) -> str:
        return " ".join(
            [self._construct_select_for_column(column) for column in self.graph.columns]
        )

    def _construct_where(self) -> str:
        triples: List[str] = []
        for edge in self.graph.edges:
            subj = self._sparql_name_for_column(
                self._column_by_node_id(edge.source.id_)
            )
            # TODO(jlscheerer) Only use the ID for now constant properties.
            pred = self._sparql_name_for_column(
                self._column_by_node_id(edge.property.id_)
            )
            obj = self._sparql_name_for_column(self._column_by_node_id(edge.target.id_))
            triples.append(f"{subj} {pred} {obj} .")

        return "\n".join(triples)

    def _construct_filter(self) -> str:
        filters: List[str] = []

        assert not self.requires_filters()

        for node in self.graph.nodes:
            if isinstance(node, QueryGraphEntityConstantNode):
                column = self._sparql_name_for_column(self._column_by_node_id(node.id_))
                filters.append(f"{column} IN ({self._construct_qid_list(node.qids)})")
            elif isinstance(node, QueryGraphPropertyNode):
                column = self._sparql_name_for_column(self._column_by_node_id(node.id_))
                filters.append(f"{column} IN ({self._construct_pid_list(node.pids)})")

        if len(filters) == 0:
            return "True"
        return " &&\n\t\t\t\t".join(filters)

    def _construct_group_by(self) -> str:
        assert self.requires_aggregation()
        group_by = []
        for column in self.graph.columns:
            if isinstance(column, AnchorEntityColumnInfo) or isinstance(
                column, PropertyColumnInfo
            ):
                group_by.append(column)
        return " ".join(map(self._sparql_name_for_column, group_by))

    def _construct_pid_list(self, pids: List[str]) -> str:
        return ", ".join([f"wdt:{pid}" for pid in pids])

    def _construct_qid_list(self, qids: List[str]) -> str:
        return ", ".join([f"wd:{qid}" for qid in qids])

    def _construct_select_for_column(self, column: ColumnInfo) -> str:
        if isinstance(column, AggregateColumnInfo):
            assert not column.aggregate.distinct
            variable = self._sparql_name_for_column(
                self._column_by_node_id(column.aggregate.source.id_)
            )
            return f"({column.aggregate.type_.name}({variable}) AS ?Z{column.aggregate.target.id_})"
        return self._sparql_name_for_column(column)

    def _sparql_name_for_column(self, column: ColumnInfo) -> str:
        if column in self.col2name:
            return self.col2name[column]

        if isinstance(column, PropertyColumnInfo):
            self.col2name[column] = f"?P{len(self.col2name)}"
        elif isinstance(column, AnchorEntityColumnInfo):
            self.col2name[column] = f"?A{len(self.col2name)}"
        elif isinstance(column, HeadVariableColumnInfo):
            self.col2name[column] = f"?H{len(self.col2name)}"
        elif isinstance(column, AggregateColumnInfo):
            self.col2name[column] = f"?G{len(self.col2name)}"
        else:
            assert False

        return self.col2name[column]

    def _dedent_query(self, query: str) -> str:
        return inspect.cleandoc(query)


def wqg2sparql(wqg: QueryGraph, emit_labels: bool = False) -> SPARQLQuery:
    assert wqg.is_executable()
    return SPARQLBackend(wqg).to_query(emit_labels)
