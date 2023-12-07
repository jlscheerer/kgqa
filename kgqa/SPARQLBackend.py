from collections import defaultdict
from enum import Enum, auto
import inspect
from dataclasses import dataclass
from typing import DefaultDict, Dict, List, Optional, Set
from typing_extensions import override

from .Config import Config
from .QueryParser import IDConstant, NumericConstant, StringConstant
from .QueryBackend import QueryBackend, QueryString
from .QueryGraph import (
    QueryGraphConstantNode,
    QueryGraphNode,
    QueryGraphPropertyConstantNode,
    QueryGraphPropertyEdgeType,
    QueryGraphPropertyNode,
    QueryGraphEntityConstantNode,
    AggregateColumnInfo,
    AnchorEntityColumnInfo,
    ColumnInfo,
    HeadVariableColumnInfo,
    PropertyColumnInfo,
    QueryGraph,
    QueryGraphVariableNode,
    VirtualEntityColumnInfo,
)


@dataclass
class SPARQLQuery(QueryString):
    pass


class PropertyOccurrence(Enum):
    WDT = auto()
    P = auto()
    PS = auto()
    PQ = auto()


class SPARQLBackend(QueryBackend):
    @override
    def to_query(self, emit_labels: bool = False) -> SPARQLQuery:
        if emit_labels:
            raise AssertionError("unsupported option 'emit_labels' for SPARQLBackend")
        type_info = str()
        config = Config()
        if config["sparql"]["emit_types"]:
            type_info = f"{self._emit_type_info()}\n"

        self._property_filter: DefaultDict[
            QueryGraphPropertyEdgeType, Set[PropertyOccurrence]
        ] = defaultdict(set)
        WHERE = self._construct_where()
        # OPTIMIZE We could omit generating FILTER(True)
        FILTER = self._construct_filter()
        GROUP_BY = str()
        if self.requires_aggregation():
            GROUP_BY = f"GROUP BY {self._construct_group_by()}"
        SELECT = self._construct_select()
        query = self._dedent_query(
            f"""{type_info}SELECT {SELECT}
WHERE
{{
    {WHERE}
    FILTER(\t{FILTER})
    SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
}}
                {GROUP_BY}"""
        )

        for column in self.col2name:
            if isinstance(column.node, QueryGraphPropertyNode) or isinstance(
                column.node, QueryGraphPropertyConstantNode
            ):
                self.col2name[column] = self._sparql_name_for_column(column)

        return SPARQLQuery(value=query, col2name=self.col2name)

    def _construct_select(self) -> str:
        return " ".join(
            [self._construct_select_for_column(column) for column in self.graph.columns]
        )

    def _construct_where(self) -> str:
        triples: List[str] = []
        for index, edge in enumerate(self.graph.edges):
            subj = self._construct_node_ref(edge.source)
            obj = self._construct_node_ref(edge.target)

            occurence: PropertyOccurrence
            if edge.source.is_qualifier() and edge.is_qualifier():
                occurence = PropertyOccurrence.PQ
            elif edge.source.is_qualifier():
                occurence = PropertyOccurrence.PS
            elif edge.target.is_qualifier():
                occurence = PropertyOccurrence.P
            else:
                occurence = PropertyOccurrence.WDT

            self._property_filter[edge.property].add(occurence)
            prop = self._construct_property_ref(edge.property, type_=occurence)

            prefix = "    " if index != 0 else ""
            triples.append(f"{prefix}{subj} {prop} {obj} .")

        return "\n".join(triples)

    def _construct_property_ref(
        self,
        property: QueryGraphPropertyEdgeType,
        type_: Optional[PropertyOccurrence] = None,
    ):
        if isinstance(property, QueryGraphPropertyConstantNode):
            # TODO(jlscheerer) We may need to adapt this for qualifiers.
            return f"wdt:{property.constant.value}"
        elif isinstance(property, QueryGraphPropertyNode):
            return self._sparql_name_for_column(
                self._column_by_node_id(property.id_), type_=type_
            )
        else:
            assert False

    def _construct_node_ref(self, node: QueryGraphNode):
        if isinstance(node, QueryGraphEntityConstantNode) or isinstance(
            node, QueryGraphVariableNode
        ):
            ref = self._sparql_name_for_column(self._column_by_node_id(node.id_))
        elif isinstance(node, QueryGraphConstantNode):
            const = node.constant
            if isinstance(const, IDConstant):
                # TODO(jlscheerer) We may need to adapt this for qualifiers.
                ref = f"wd:{const.value}"
            elif isinstance(const, StringConstant):
                ref = f'"{const.value}"'
            elif isinstance(const, NumericConstant):
                # NOTE Apparently, this is allowed: ?X wdt:P2046 22.1 .
                ref = f"{const.value}"
            else:
                assert False
        else:
            assert False
        return ref

    def _construct_filter(self) -> str:
        filters: List[str] = []

        assert not self.requires_filters()

        for node in self.graph.nodes:
            if isinstance(node, QueryGraphEntityConstantNode):
                column = self._sparql_name_for_column(self._column_by_node_id(node.id_))
                filters.append(f"{column} IN ({self._construct_qid_list(node.qids)})")
            elif isinstance(node, QueryGraphPropertyNode):
                ref: str
                if PropertyOccurrence.WDT in self._property_filter[node]:
                    column = self._sparql_name_for_column(
                        self._column_by_node_id(node.id_)
                    )
                    filters.append(
                        f"{column} IN ({self._construct_pid_list(node.pids)})"
                    )
                    ref = self._construct_iri_ref(column, PropertyOccurrence.WDT)
                elif PropertyOccurrence.P in self._property_filter[node]:
                    column = self._sparql_name_for_column(
                        self._column_by_node_id(node.id_), type_=PropertyOccurrence.P
                    )
                    filters.append(
                        f"{column} IN ({self._construct_pid_list(node.pids, prefix='p')})"
                    )
                    ref = self._construct_iri_ref(column, PropertyOccurrence.P)
                else:
                    assert len(self._property_filter[node]) == 1
                    assert PropertyOccurrence.PQ in self._property_filter[node]
                    column = self._sparql_name_for_column(
                        self._column_by_node_id(node.id_), type_=PropertyOccurrence.PQ
                    )
                    filters.append(
                        f"{column} IN ({self._construct_pid_list(node.pids, prefix='pq')})"
                    )

                    # NOTE we don't need to enforce equality constraints on the refs as there are none.
                    continue

                for type_ in self._property_filter[node]:
                    if type_ == PropertyOccurrence.WDT:
                        continue
                    if (
                        type_ == PropertyOccurrence.P
                        and PropertyOccurrence.WDT not in self._property_filter[node]
                    ):
                        continue

                    filters.append(
                        f"{self._construct_iri_ref(self._sparql_name_for_column(self._column_by_node_id(node.id_), type_=type_), type_)} = {ref}"
                    )

        if len(filters) == 0:
            return "True"
        return " &&\n\t\t".join(filters)

    def _construct_iri_ref(self, column: str, type_: PropertyOccurrence):
        prefix = f"{type_.name.lower()}:"
        return f"strafter(str({column}), str({prefix}))"

    def _construct_group_by(self) -> str:
        assert self.requires_aggregation()
        group_by = []
        for column in self.graph.columns:
            if isinstance(column, AnchorEntityColumnInfo) or isinstance(
                column, PropertyColumnInfo
            ):
                group_by.append(column)
        return " ".join(map(self._sparql_name_for_column, group_by))

    def _construct_pid_list(self, pids: List[str], prefix="wdt") -> str:
        return ", ".join([f"{prefix}:{pid}" for pid in pids])

    def _construct_qid_list(self, qids: List[str]) -> str:
        return ", ".join([f"wd:{qid}" for qid in qids])

    def _construct_select_for_column(self, column: ColumnInfo) -> str:
        if isinstance(column, AggregateColumnInfo):
            assert not column.aggregate.distinct
            variable = self._sparql_name_for_column(
                self._column_by_node_id(column.aggregate.source.id_)
            )
            # TODO(jlscheerer) We should move Z_ to _sparql_name_for_column
            aggregate_var = self._sparql_name_for_column(
                self._column_by_node_id(column.aggregate.target.id_)
            )
            return f"({column.aggregate.type_.name}({variable}) AS {aggregate_var})"
        return self._sparql_name_for_column(column)

    def _sparql_name_for_column(
        self, column: ColumnInfo, type_: Optional[PropertyOccurrence] = None
    ) -> str:
        assert (type_ is None) or isinstance(column, PropertyColumnInfo)
        if column in self.col2name:
            if isinstance(column, PropertyColumnInfo):
                if type_ is None:
                    if PropertyOccurrence.WDT in self._property_filter[column.node]:
                        type_ = PropertyOccurrence.WDT
                    elif PropertyOccurrence.P in self._property_filter[column.node]:
                        type_ = PropertyOccurrence.P
                    else:
                        assert (
                            PropertyOccurrence.PQ in self._property_filter[column.node]
                        )
                        type_ = PropertyOccurrence.PQ
                raw = self.col2name[column][2:]
                if type_ == PropertyOccurrence.P:
                    return f"?P{raw}"
                if type_ == PropertyOccurrence.PS:
                    return f"?PS{raw}"
                if type_ == PropertyOccurrence.PQ:
                    return f"?PQ{raw}"
            return self.col2name[column]

        if isinstance(column, PropertyColumnInfo):
            self.col2name[column] = f"?C{len(self.col2name)}"
            return self._sparql_name_for_column(column, type_)
        elif isinstance(column, AnchorEntityColumnInfo):
            self.col2name[column] = f"?A{len(self.col2name)}"
        elif isinstance(column, HeadVariableColumnInfo):
            self.col2name[column] = f"?H{len(self.col2name)}"
        elif isinstance(column, AggregateColumnInfo):
            self.col2name[column] = f"?G{len(self.col2name)}"
        elif isinstance(column, VirtualEntityColumnInfo):
            self.col2name[column] = f"?V{len(self.col2name)}"
        else:
            assert False

        return self.col2name[column]

    def _emit_type_info(self) -> str:
        types = []
        for node in self.graph.nodes:
            if isinstance(node, QueryGraphVariableNode):
                column = self._sparql_name_for_column(self._column_by_node_id(node.id_))
                types.append(f"#pragma:type {column} {node.variable.type_info()}")
        return "\n".join(types)

    def _dedent_query(self, query: str) -> str:
        return inspect.cleandoc(query)


def wqg2sparql(wqg: QueryGraph, emit_labels: bool = False) -> SPARQLQuery:
    assert wqg.is_executable()
    return SPARQLBackend(wqg).to_query(emit_labels)
