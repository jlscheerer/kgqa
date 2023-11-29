import abc
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple

from kgqa.QueryGraph import (
    AggregateColumnInfo,
    AnchorEntityColumnInfo,
    ColumnInfo,
    EntityColumnInfo,
    ExecutableQueryGraph,
    HeadEntityColumnInfo,
    PropertyColumnInfo,
    QueryGraphEdge,
    QueryGraphId,
    QueryStatistics,
)
from kgqa.QueryParser import Aggregation, Variable


@dataclass
class VariableEdgeOccurrence:
    edge_index: int
    is_subj: bool


@dataclass
class QueryString:
    value: str


class QueryBackend(abc.ABC):
    graph: ExecutableQueryGraph

    # Tuples consisting of Subject, Object, Edge
    edge_list: List[Tuple[QueryGraphId, QueryGraphId, QueryGraphEdge]]

    # Maps each variable to the edges it occurs in.
    var2edges: Dict[QueryGraphId, List[VariableEdgeOccurrence]]

    columns: List[ColumnInfo]
    aggregate_columns: List[HeadEntityColumnInfo]

    def __init__(self, wqg: ExecutableQueryGraph):
        self.graph = wqg
        self.edge_list = [
            (subj_id, obj_id, edge)
            for (subj_id, obj_id), edges in wqg.edges.items()
            for edge in edges
        ]

        # Construct a mapping from each var to edge_id and position.
        var2edges = defaultdict(list)
        for index, (subj_id, obj_id, _) in enumerate(self.edge_list):
            var2edges[subj_id].append(
                VariableEdgeOccurrence(edge_index=index, is_subj=True)
            )
            var2edges[obj_id].append(
                VariableEdgeOccurrence(edge_index=index, is_subj=False)
            )
        self.var2edges = dict(var2edges)

        # Construct the required output columns: Properties, Anchors and Head Variables
        self.columns = []
        for index, (_, _, edge) in enumerate(self.edge_list):
            self.columns.append(
                PropertyColumnInfo(index=index, predicate=edge.predicate)
            )

        for node in self.graph.nodes:
            if not node.is_free:
                self.columns.append(
                    AnchorEntityColumnInfo(index=node.id_, entity=node.value)
                )

        aggregate_index = 0
        for head_var in self.graph.head:
            if isinstance(head_var, Variable):
                head_var_id = self.graph.arg2id[head_var]
                self.columns.append(
                    HeadEntityColumnInfo(index=head_var_id, entity=head_var)
                )
            elif isinstance(head_var, Aggregation):
                # Create a "virtual column" for variables in aggregates.
                # TODO(jlscheerer) This is not supported by the "native" SQL-Backend.
                head_var_id = self.graph.arg2id[head_var.var]
                self.columns.append(
                    AggregateColumnInfo(
                        index=head_var_id,
                        entity=head_var.var,
                        type_=head_var.type_,
                        distinct=False,
                        aggregate_index=aggregate_index,
                    )
                )
                aggregate_index += 1
            else:
                assert False

    @abc.abstractmethod
    def to_query(
        self, stats: QueryStatistics, emit_labels: bool = False
    ) -> QueryString:
        """
        Generates an executable query string through the QueryBackend.
        """
        pass

    def requires_filters(self) -> bool:
        return len(self.graph.filter_var_ids) != 0

    def requires_aggregation(self) -> bool:
        raise AssertionError()

    def _column_by_node_id(self, node_id: QueryGraphId) -> EntityColumnInfo:
        # TODO(jlscheerer) We can definitely improve this.
        for column in self.columns:
            if isinstance(column, EntityColumnInfo):
                if column.index == node_id:
                    return column
            elif isinstance(column, AggregateColumnInfo):
                if column.index == node_id:
                    return column
        raise AssertionError("attempting to get column info for invalid node")

    def _column_by_edge_index(self, index: int) -> PropertyColumnInfo:
        # TODO(jlscheerer) We can definitely improve this.
        for column in self.columns:
            if isinstance(column, PropertyColumnInfo):
                if column.index == index:
                    return column
        raise AssertionError("attempting to get column info for invalid edge")
