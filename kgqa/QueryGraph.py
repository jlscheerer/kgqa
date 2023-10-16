import abc
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Set, Tuple, cast
from typing_extensions import override

from kgqa.MatchingUtils import match_entities, match_predicates

from .QueryParser import (
    ArgumentType,
    ParsedQuery,
    PredicateType,
    QueryFilter,
    Variable,
)


@dataclass
class QueryStatistics:
    # TODO(jlscheerer) We need to extend this class to match the old behavior
    pid2scores: Dict[Any, Any] = field(default_factory=dict)
    qid2scores: Dict[Any, Any] = field(default_factory=dict)

    def set_scores(self, pid2scores, qid2scores) -> None:
        self.pid2scores = pid2scores
        self.qid2scores = qid2scores


@dataclass
class QueryGraphId:
    value: int

    def __eq__(self, other) -> bool:
        if not isinstance(other, QueryGraphId):
            return False
        return self.value == other.value

    def __hash__(self):
        return hash(self.value)


class QueryGraphIdGenerator:
    _current_id: int = 0

    def __call__(self):
        id_ = self._current_id
        self._current_id += 1
        return QueryGraphId(value=id_)


@dataclass
class QueryGraphNode:
    id_: QueryGraphId
    is_free: bool
    value: ArgumentType


@dataclass
class QueryGraphEdge:
    predicate: PredicateType


@dataclass
class QueryGraph(abc.ABC):
    arg2id: Dict[ArgumentType, QueryGraphId]

    nodes: List[QueryGraphNode]
    # NOTE: Fixes a previous issues where two nodes could not share multiple predicates.
    edges: Dict[Tuple[QueryGraphId, QueryGraphId], List[QueryGraphEdge]]

    head_var_ids: List[QueryGraphId]
    filter_var_ids: Set[QueryGraphId]

    filters: List[QueryFilter]
    anchors: List[ArgumentType]

    def is_cyclic(self):
        return len(self.edges) == len(self.nodes)

    def is_acyclic(self):
        return not self.is_cyclic()

    def is_abstract(self) -> bool:
        return False

    def is_executable(self) -> bool:
        return False


@dataclass
class AbstractQueryGraph(QueryGraph):
    @override
    def is_abstract(self) -> bool:
        return True


@dataclass
class ExecutableQueryGraph(QueryGraph):
    @override
    def is_executable(self) -> bool:
        return True


def _construct_aqg_from_pq(pq: ParsedQuery) -> AbstractQueryGraph:
    """
    Construct a query graph from a parsed query.
    The type of the constructed graph is always USER.
    """
    make_id = QueryGraphIdGenerator()

    # Free variables and anchors to IDs, i.e., any argument to a predicate.
    arg2id: Dict[ArgumentType, QueryGraphId] = dict()

    nodes: List[QueryGraphNode] = []
    predicates: List[PredicateType] = []
    anchors: List[ArgumentType] = []
    for subj, pred, obj in pq.spo():
        predicates.append(pred)
        for entity in [subj, obj]:
            if not isinstance(entity, Variable):
                # Any constant is considered an anchor for now.
                anchors.append(entity)

            if entity not in arg2id:
                new_id = make_id()
                arg2id[entity] = new_id
                nodes.append(
                    QueryGraphNode(
                        id_=new_id, is_free=isinstance(entity, Variable), value=entity
                    )
                )

    filter_var_ids = set()
    for filter in pq.filters:
        filter_var_ids.add(arg2id[filter.lhs])

    edges = defaultdict(list)  # type: ignore
    for subj, pred, obj in pq.spo():
        i, j = arg2id[subj], arg2id[obj]
        edges[(i, j)].append(QueryGraphEdge(predicate=pred))

    # TODO(jlscheerer) Assumes head only contains vars, i.e., no Aggregation support
    head_var_ids = [arg2id[x] for x in pq.head]  # type: ignore

    return AbstractQueryGraph(
        arg2id=arg2id,
        nodes=nodes,
        edges=dict(edges),
        head_var_ids=head_var_ids,
        filter_var_ids=filter_var_ids,
        filters=pq.filters,
        anchors=cast(List[ArgumentType], predicates) + anchors,
    )


def query2aqg(pq: ParsedQuery) -> Tuple[AbstractQueryGraph, QueryStatistics]:
    return _construct_aqg_from_pq(pq), QueryStatistics()


def aqg2wqg(
    aqg: AbstractQueryGraph, stats: QueryStatistics
) -> Tuple[ExecutableQueryGraph, QueryStatistics]:
    wqg = ExecutableQueryGraph(
        arg2id=deepcopy(aqg.arg2id),
        nodes=deepcopy(aqg.nodes),
        edges=deepcopy(aqg.edges),
        head_var_ids=deepcopy(aqg.head_var_ids),
        filter_var_ids=deepcopy(aqg.filter_var_ids),
        filters=deepcopy(aqg.filters),
        anchors=deepcopy(aqg.anchors),
    )

    # Transform edges of the graph.
    pid2scores = match_predicates(aqg)

    # Transform entities of the graph
    qid2scores = match_entities(aqg)

    # Add scores info the stats.
    stats.set_scores(pid2scores, qid2scores)

    return wqg, stats
