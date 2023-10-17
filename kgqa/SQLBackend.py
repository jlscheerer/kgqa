from dataclasses import dataclass
from typing import Dict, Tuple, List

from .QueryGraph import (
    ExecutableQueryGraph,
    QueryGraphEdge,
    QueryGraphId,
    QueryGraphNode,
    QueryStatistics,
)


@dataclass
class SQL_IR:
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

    def to_sql(self, query_stats: QueryStatistics, emit_labels: bool = False) -> str:
        SELECT, nums_and_cols_stats = self._construct_select()
        query_stats.add_nums_and_cols(*nums_and_cols_stats)
        FROM = self._construct_from()
        WHERE = self._construct_where()
        query = f"""SELECT {SELECT} FROM {FROM} WHERE {WHERE}"""
        if emit_labels:
            query = self._join_labels(query, query_stats.column_names)
        else:
            query = f"{query};"
        return query

    def _construct_from(self) -> str:
        # NOTE We cut out some logic related to templated QueryGraphs.
        claims_relname = "claims_5m_inv"
        claims = [f"{claims_relname} c{i}" for i in range(0, len(self.edge_list))]
        return ", ".join(claims)

    def _get_all_sql_address_for_var(self, varid: QueryGraphId) -> List[str]:
        adrs: List[str] = []
        # Check if current varid is going to be a filter.
        # If that's the case, there must be 1 edge (we make it non-joinable for now.)
        # Then, this edge must be going out of something, and endpoint must be "datavalue_string"
        object_type = "datavalue_entity"
        if varid in self.graph.filter_var_ids:
            assert len(self.var2edges[varid]) == 1
            # Todo: other object types?
            object_type = "datavalue_string"
        all_addrs_raw = self.var2edges[varid]
        for edge_id, is_subj in all_addrs_raw:
            so_type = "entity_id" if is_subj else object_type
            adrs.append(f"c{edge_id}.{so_type}")
        return adrs

    def _construct_select(self) -> Tuple[str, Tuple[int, int, int, List[str]]]:
        # NOTE We cut out some logic related to templated QueryGraphs.
        heads = []
        col_names = []

        # Select all properties (predicates)
        for i, (_, edges) in enumerate(self.graph.edges.items()):
            # TODO(jlscheerer) Update this assertion, Mimic legacy behavior.
            assert len(edges) == 1
            edge = edges[0]

            col_name = f"{edge.predicate.query_name()} ({i}) (pid)"
            heads.append(f'c{i}.property AS "{col_name}"')
            col_names.append(col_name)

        # Select all anchors
        num_anchors = 0
        for node in self.graph.nodes:
            if not node.is_free:
                node_addr = self._get_all_sql_address_for_var(node.id_)[0]
                col_name = f"{node.to_str()} (qid)"
                heads.append(f'{node_addr} AS "{col_name}"')
                col_names.append(col_name)
                num_anchors += 1

        # Select all head free vars.
        for i, hv_id in enumerate(self.graph.head_var_ids):
            adrs = self._get_all_sql_address_for_var(hv_id)
            # We are free to use any address for this var when it's a head.
            col_name = self.graph.nodes[hv_id.value].to_str()
            heads.append(f'{adrs[0]} AS "{col_name}"')
            col_names.append(col_name)

        nums_and_cols_stats = (
            len(self.graph.edges),
            num_anchors,
            len(self.graph.head_var_ids),
            col_names,
        )
        return ", ".join(heads), nums_and_cols_stats

    def _construct_where_join_condition(
        self, where_conds: List[str], all_adrs: List[str]
    ) -> None:
        all_adrs += [all_adrs[0]]
        cond = [f"{all_adrs[i]} = {all_adrs[i+1]}" for i in range(len(all_adrs) - 1)]
        where_conds.append(" AND ".join(cond))

    def _construct_pid_list(self, pid_list: List[str]) -> str:
        pid_list_single_quote = [f"'{x}'" for x in pid_list]
        return ", ".join(pid_list_single_quote)

    def _construct_where_anchor_node(
        self, node: QueryGraphNode, where_conds: List[str]
    ) -> None:
        node_addr = self._get_all_sql_address_for_var(node.id_)[0]
        qids = self.graph.matched_anchors_qids[node.id_]
        qids_list = self._construct_pid_list(qids)
        where_conds.append(f"{node_addr} IN ({qids_list})")

    def _construct_where(self) -> str:
        # NOTE We cut out some logic related to templated QueryGraphs.
        where_conds: List[str] = []
        for i, (_, edges) in enumerate(self.edge_list):
            # TODO(jlscheerer) Update this assertion, mimic legacy behavior.
            assert len(edges) == 1
            edge = edges[0]

            where_conds.append(
                f"c{i}.property IN ({self._construct_pid_list(edge.get_matched_pids())})"
            )
        for node in self.graph.nodes:
            all_adrs = self._get_all_sql_address_for_var(node.id_)
            if len(all_adrs) >= 2:
                self._construct_where_join_condition(where_conds, all_adrs)
            if not node.is_free:
                self._construct_where_anchor_node(node, where_conds)
        # Add filtering conditions.
        # TODO(jlscheerer) Support filtering conditions here.
        # for filter_ in self.graph.filters:
        #     filter_id = self.graph.var2ids[filter_[0]]
        #     addr = self.get_all_sql_address_for_var(filter_id)[0]
        #     where_conds.append(
        #         f"substring({addr}, '\\d+')::integer {filter_[1]} {filter_[2]}"
        #     )
        return " AND ".join(where_conds)

    def _strip_qid_if_needed(self, column_name: str) -> str:
        if len(column_name) > 6 and (
            column_name[-5:] == "(pid)" or column_name[-5:] == "(qid)"
        ):
            return column_name[:-6]
        return column_name

    def _join_labels(self, query: str, column_names: List[str]) -> str:
        selections = ", ".join(
            [
                f'oq."{column}", l{index}.value AS "{self._strip_qid_if_needed(column)} (label)"'
                for (index, column) in enumerate(column_names)
            ]
        )
        label_joins = " ".join(
            [
                f'LEFT JOIN labels_en l{index} ON l{index}.id = oq."{column}"'
                for (index, column) in enumerate(column_names)
            ]
        )
        return f"""WITH orig_query AS ({query})
                   SELECT {selections}
                   FROM orig_query oq {label_joins};"""


def wqg2sql(
    wqg: ExecutableQueryGraph, query_stats: QueryStatistics, emit_labels: bool = False
) -> Tuple[str, QueryStatistics]:
    sql = SQL_IR(wqg).to_sql(query_stats, emit_labels)
    return sql, query_stats
