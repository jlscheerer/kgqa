import abc
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple

from kgqa.QueryGraph import (
    AnchorEntityColumnInfo,
    ColumnInfo,
    ExecutableQueryGraph,
    HeadEntityColumnInfo,
    PropertyColumnInfo,
    QueryGraphEdge,
    QueryGraphId,
    QueryStatistics,
)
from kgqa.QueryParser import Variable


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

        for head_var_id in self.graph.head_var_ids:
            # NOTE As entity occurs in the head of the query, it must be a variable.
            entity = self.graph.nodes[head_var_id.value].value
            assert isinstance(entity, Variable)

            self.columns.append(HeadEntityColumnInfo(index=head_var_id, entity=entity))

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

    def _column_by_edge_index(self, index: int) -> PropertyColumnInfo:
        # TODO(jlscheerer) We can definitely improve this.
        for column in self.columns:
            if isinstance(column, PropertyColumnInfo):
                if column.index == index:
                    return column
        raise AssertionError("attempting to get column info for invalid edge")
