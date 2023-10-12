from copy import deepcopy
from enum import Enum

from .QueryParser import QueryParser, ParsedQuery, is_var
from .MatchingUtils import match_predicates, match_entities, Edge


class QueryGraphType(Enum):
    USER =       1  # truly abstract graph, just parsed from NL-FOL query
    EXECUTABLE = 2  # stands for a graph that is ready to be turned into SQL on wikidata, verbatim
    TEMPLATED =  3  # stands for a graph template for which we need to sample certain preds/entities


class QueryStatistics:
    """
    Responsible for keeping track of
    1) for each predicate, mapping from matches to pids
    2) for each entity, mapping from matches to qids
    """

    def __init__(self):
        # Query statistics.
        self.num_preds = -1
        self.num_anchors = -1
        self.num_heads = -1
        self.col_names = None

        # Populated after matching.
        # Each will store: { ?? -> {[p/q]id -> score}}
        self.p2s = None
        self.q2s = None

    def add_nums_and_cols(self, np, na, nh, cn):
        self.num_preds = np
        self.num_anchors = na
        self.num_heads = nh
        self.col_names = cn

    def set_pid_qid_scores(self, p2s, q2s):
        self.p2s = p2s
        self.q2s = q2s

    def get_number_of_matched_cols(self):
        return self.num_preds + self.num_anchors


class Node:
    def __init__(self, id_, is_free, value):
        self.id = id_
        self.is_free = is_free
        self.value = value

    def get_str(self):
        if type(self.value) == str:
            return self.value
        elif is_var(self.value):
            return self.value.get_str()
        else:
            print(f"strange value! {self.value}")


def construct_templated_AQG(vars_ids, edges, ground_var_ids):
    """
    Construct a templated query graph from vars & edges. Type is TEMPLATED.
    """
    type_ = QueryGraphType.TEMPLATED
    var2ids = {var_: id_ for (id_, var_) in vars_ids}
    nodes = [
        Node(id_, id_ not in ground_var_ids, "" if id_ in ground_var_ids else var_)
        for (id_, var_) in vars_ids
    ]
    edges = {e: Edge("") for e in edges}
    head_vars_ids = [id_ for (id_, _) in vars_ids]
    return AbstractQueryGraph(type_, var2ids, nodes, edges, head_vars_ids, [], [])


def construct_AQG_from_PQ(pq: ParsedQuery):
    """
    Construct graph from a parsed query. Type is always USER.
    """
    type_ = QueryGraphType.USER

    # free_vars & anchors to ids mapping
    var2ids = dict()

    # list of all nodes
    nodes = []
    all_user_preds = []
    all_user_anchors = []
    for s, p, o in pq.clauses:
        all_user_preds.append(p)
        for sss in [s, o]:
            if not is_var(sss):
                all_user_anchors.append(sss)
            if sss not in var2ids.keys():
                var2ids[sss] = len(nodes)
                nodes.append(Node(len(nodes), is_var(sss), sss))

    filter_var_ids = set()
    for var, _, _ in pq.filters:
        filter_var_ids.add(var2ids[var])
    filter_var_ids = list(filter_var_ids)

    n = len(nodes)
    edges = dict()

    for s, p, o in pq.clauses:
        i, j = var2ids[s], var2ids[o]
        edges[(i, j)] = Edge(p)

    head_vars_ids = [var2ids[x] for x in pq.head]

    aqg = AbstractQueryGraph(
        type_,
        var2ids,
        nodes,
        edges,
        head_vars_ids,
        filter_var_ids,
        pq.filters,
        pq.expressq,
    )
    aqg.all_user_preds_anchors = all_user_preds + all_user_anchors
    return aqg


class AbstractQueryGraph:
    def __init__(
        self,
        type_,
        var2ids,
        nodes,
        edges,
        head_vars_ids,
        filter_var_ids,
        filters,
        orig_query="",
    ):
        self.type_ = type_
        self.var2ids = var2ids  # {str/Var -> int}
        self.nodes = nodes  # [Node]
        self.edges = edges  # (i,j) -> Edge
        self.head_vars_ids = head_vars_ids  # [int]
        self.filter_var_ids = filter_var_ids  # [int]
        self.filters = filters
        self.anchors_to_wiki = dict()  # {int(id) -> [str (qids)]}

        self.all_user_preds_anchors = list()  # list of strings

        self.orig_query = orig_query

    def __repr__(self):
        nodes = [
            f'"{x.value}"' if isinstance(x.value, str) else x.value.v
            for x in self.nodes
        ]
        edges = [f"{k}:{v}" for k, v in self.edges.items()]
        return ",".join(nodes) + "\n" + ",".join(edges)

    def get_n_edges(self):
        return len(self.edges)

    def is_cyclic(self):
        return len(self.edges) == len(self.nodes)

    def is_acyclic(self):
        return not self.is_cyclic()

    def is_user(self):
        return self.type_ == QueryGraphType.USER

    def is_executable(self):
        return self.type_ == QueryGraphType.EXECUTABLE

    def is_templated(self):
        return self.type_ == QueryGraphType.TEMPLATED

    def is_ready_for_sqlize(self):
        return self.is_executable() or self.is_templated()

    def mark_executable(self):
        self.type_ = QueryGraphType.EXECUTABLE


def query2aqg(pq: ParsedQuery):
    return construct_AQG_from_PQ(pq), QueryStatistics()


def aqg2wqg(aqg, query_stats):
    """
    Given an abstract query graph, we will create a new AQG with the following conditions:
    1) each edge is represented as list of options of top-k predicates.
    2) self.anchors_to_wiki is populated, so that self.anchors_to_wiki['anchor_str'] = ['Q123', 'Q345', ...]
    """
    wqg = deepcopy(aqg)

    ### EDGES TRANSFORM ###
    pid_2_scores = match_predicates(wqg)

    ### ENTITIES TRANSFORM ###
    qid_2_scores = match_entities(wqg)

    # Add scores info to query_stats.
    query_stats.set_pid_qid_scores(pid_2_scores, qid_2_scores)

    wqg.mark_executable()
    return wqg, query_stats
