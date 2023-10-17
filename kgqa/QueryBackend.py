import abc
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


class QueryBackend(abc.ABC):
    graph: ExecutableQueryGraph
    edge_list: List[Tuple[Tuple[QueryGraphId, QueryGraphId], List[QueryGraphEdge]]]

    # TODO(jlscheerer) Refactor this.
    # Maps each node to the list of edges it occurs in. (index_of_edge, is_subj)
    var2edges: Dict[QueryGraphId, List[Tuple[int, int]]]

    columns: List[ColumnInfo]

    def __init__(self, wqg: ExecutableQueryGraph):
        self.graph = wqg

        # TODO(jlscheerer) We probably want to flaten the inner edges.
        self.edge_list = [(edge_id, edges) for edge_id, edges in wqg.edges.items()]

        # Construct a mapping from each var to edge_id and position.
        self.var2edges: Dict[QueryGraphId, List[Tuple[int, int]]] = dict()
        for index, ((subj_id, obj_id), _) in enumerate(self.edge_list):
            self.var2edges[subj_id] = self.var2edges.get(subj_id, []) + [(index, 1)]
            self.var2edges[obj_id] = self.var2edges.get(obj_id, []) + [(index, 0)]

        # Construct the required output columns: Properties, Anchors and Head Variables
        self.columns = []
        for index, (_, edges) in enumerate(self.graph.edges.items()):
            # TODO(jlscheerer) Temporary assumption to mimic legacy behavior
            assert len(edges) == 1
            edge = edges[0]

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
    def to_query(self, stats: QueryStatistics, emit_labels: bool = False) -> str:
        pass
