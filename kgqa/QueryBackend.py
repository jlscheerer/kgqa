import abc
from dataclasses import dataclass
from typing import Dict, List, Tuple
from typing_extensions import override

from kgqa.QueryGraph import (
    ExecutableQueryGraph,
    QueryGraphEdge,
    QueryGraphId,
    QueryStatistics,
)
from kgqa.QueryParser import (
    ArgumentType,
    IDConstant,
    PredicateType,
    StringConstant,
    Variable,
)


@dataclass
class ColumnInfo(abc.ABC):
    @abc.abstractmethod
    def __repr__(self) -> str:
        """
        Constructs a serializable name for the column.
        """
        pass


@dataclass
class PropertyColumnInfo(ColumnInfo):
    index: int  # TODO(jlscheerer) We should probably abstract this.
    predicate: PredicateType

    @override
    def __repr__(self) -> str:
        return f"{self.predicate.query_name()} ({self.index}) (pid)"


@dataclass
class EntityColumnInfo(ColumnInfo):
    entity: ArgumentType

    @override
    def __repr__(self) -> str:
        if isinstance(self.entity, StringConstant):
            return f"{self.entity.value} (qid)"
        elif isinstance(self.entity, Variable):
            return f"Variable({self.entity.name}) (qid)"
        elif isinstance(self.entity, IDConstant):
            return f"IDConstant({self.entity.value}) (qid)"

        # TODO(jlscheerer) Eventually we need to handle different types here.
        assert False


@dataclass
class HeadEntityColumnInfo(EntityColumnInfo):
    """
    Entity that occurs in the head of a query.
    """

    entity: Variable

    @override
    def __repr__(self) -> str:
        return f"Variable({self.entity.name})"


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
