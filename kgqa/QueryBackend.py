import abc
from typing import Dict, List, Tuple

from kgqa.QueryGraph import (
    ExecutableQueryGraph,
    QueryGraphEdge,
    QueryGraphId,
    QueryStatistics,
)


class QueryBackend(abc.ABC):
    graph: ExecutableQueryGraph
    edge_list: List[Tuple[Tuple[QueryGraphId, QueryGraphId], List[QueryGraphEdge]]]
    var2edges: Dict[QueryGraphId, List[Tuple[int, int]]]

    def __init__(self, wqg: ExecutableQueryGraph):
        self.graph = wqg

        # TODO(jlscheerer) We probably want to flaten the inner edges.
        self.edge_list = [(edge_id, edges) for edge_id, edges in wqg.edges.items()]

        # Construct a mapping from each var to edge_id and position.
        self.var2edges: Dict[QueryGraphId, List[Tuple[int, int]]] = dict()
        for index, ((subj_id, obj_id), _) in enumerate(self.edge_list):
            self.var2edges[subj_id] = self.var2edges.get(subj_id, []) + [(index, 1)]
            self.var2edges[obj_id] = self.var2edges.get(obj_id, []) + [(index, 0)]

    @abc.abstractmethod
    def to_query(self, stats: QueryStatistics, emit_labels: bool = False) -> str:
        pass
