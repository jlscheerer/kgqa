import abc
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Set, Tuple, cast, reveal_type
from typing_extensions import override

from kgqa.MatchingUtils import compute_similar_entity_ids, compute_similar_predicates

from .QueryParser import (
    ArgumentType,
    IDConstant,
    ParsedQuery,
    PredicateType,
    QueryFilter,
    StringConstant,
    Variable,
)


@dataclass
class QueryStatistics:
    # TODO(jlscheerer) We need to extend this class to match the old behavior
    pid2scores: Dict[Any, Any] = field(default_factory=dict)
    qid2scores: Dict[Any, Any] = field(default_factory=dict)

    num_predicates: int = -1
    num_anchors: int = -1
    num_heads: int = -1
    column_names: List[str] = field(default_factory=list)

    def set_scores(self, pid2scores, qid2scores) -> None:
        self.pid2scores = pid2scores
        self.qid2scores = qid2scores

    def add_nums_and_cols(
        self,
        num_predicates: int,
        num_anchors: int,
        num_heads: int,
        column_names: List[str],
    ) -> None:
        self.num_predicates = num_predicates
        self.num_anchors = num_anchors
        self.num_heads = num_heads
        self.column_names = column_names


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

    def to_str(self) -> str:
        if isinstance(self.value, StringConstant):
            return self.value.value
        elif isinstance(self.value, Variable):
            return f"Variable({self.value.name})"
        elif isinstance(self.value, IDConstant):
            return f"IDConstant({self.value.value})"

        # TODO(jlscheerer) Eventually we need to handle different types here.
        assert False


@dataclass
class QueryGraphEdge:
    predicate: PredicateType
    matched_pids: List[str] = field(default_factory=list)

    def set_matched_pids(self, pids: List[str]) -> None:
        self.matched_pids = pids

    def get_matched_pids(self) -> List[str]:
        return self.matched_pids


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
    matched_anchors_qids: Dict[QueryGraphId, List[str]] = field(default_factory=dict)

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


def _match_predicates(wqg: ExecutableQueryGraph):
    pred_to_pid_to_score = dict()
    for _, edges in wqg.edges.items():
        for edge in edges:
            if isinstance(edge.predicate, IDConstant):
                # For IDConstants there is not much to do.
                pids, scores = [edge.predicate.value], [1.0]
            else:
                assert isinstance(edge.predicate, Variable)
                pids, scores = compute_similar_predicates(edge.predicate.query_name())
            scores = [round(x, 2) for x in scores]
            pred_to_pid_to_score[edge.predicate.query_name()] = dict(zip(pids, scores))
            edge.set_matched_pids(pids)
            # TODO(jlscheerer) Reimplement follow up via LanguageModel here.
    return pred_to_pid_to_score


def _match_entities(wqg: ExecutableQueryGraph):
    ent_to_qid_to_score = dict()
    for node in wqg.nodes:
        if not node.is_free:
            # Temporary assumption. TODO(jlscheerer) handle different constants.
            if isinstance(node.value, StringConstant):
                qids, scores = compute_similar_entity_ids(node.value.value)
            elif isinstance(node.value, IDConstant):
                qids, scores = [node.value.value], [1.0]
            else:
                assert False
            scores = [round(x, 2) for x in scores]
            wqg.matched_anchors_qids[node.id_] = qids
            ent_to_qid_to_score[node.value.value] = dict(zip(qids, scores))
    return ent_to_qid_to_score


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
    pid2scores = _match_predicates(wqg)

    # Transform entities of the graph
    qid2scores = _match_entities(wqg)

    # Add scores info the stats.
    stats.set_scores(pid2scores, qid2scores)

    return wqg, stats
