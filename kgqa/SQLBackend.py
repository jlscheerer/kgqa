from typing import Tuple, List
from typing_extensions import override

from .QueryBackend import (
    QueryBackend,
)
from .QueryGraph import (
    AnchorEntityColumnInfo,
    ColumnInfo,
    ExecutableQueryGraph,
    HeadEntityColumnInfo,
    PropertyColumnInfo,
    QueryGraphId,
    QueryGraphNode,
    QueryStatistics,
)


class SQLBackend(QueryBackend):
    @override
    def to_query(self, stats: QueryStatistics, emit_labels: bool = False) -> str:
        SELECT = self._construct_select()
        FROM = self._construct_from()
        WHERE = self._construct_where()
        query = f"""SELECT {SELECT} FROM {FROM} WHERE {WHERE}"""

        stats.set_column_info(self.columns)
        if emit_labels:
            query = self._join_query_labels(query, stats.columns)
        else:
            query = f"{query};"

        return query

    def _construct_from(self) -> str:
        # NOTE We cut out some logic related to templated QueryGraphs.
        claims_relname = "claims_5m_inv"
        claims = [f"{claims_relname} c{i}" for i in range(0, len(self.edge_list))]
        return ", ".join(claims)

    def _get_all_sql_address_for_var(self, varid: QueryGraphId) -> List[str]:
        # TODO(jlscheerer) Refactor this code.
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

    def _construct_select(self) -> str:
        # NOTE We cut out some logic related to templated QueryGraphs.
        heads: List[str] = []
        for column in self.columns:
            if isinstance(column, PropertyColumnInfo):
                heads.append(f'c{column.index}.property AS "{column}"')
            elif isinstance(column, AnchorEntityColumnInfo):
                # NOTE for the purpose of the select, it does not matter what column we
                #      reference, so we just pick the first.
                node_addr = self._get_all_sql_address_for_var(column.index)[0]
                heads.append(f'{node_addr} AS "{column}"')
            else:
                assert isinstance(column, HeadEntityColumnInfo)
                # NOTE for the purpose of the select, it does not matter what column we
                #      reference, so we just pick the first.
                node_addr = self._get_all_sql_address_for_var(column.index)[0]
                heads.append(f'{node_addr} AS "{column}"')
        return ", ".join(heads)

    def _construct_where_join_condition(
        self, where_conds: List[str], all_adrs: List[str]
    ) -> None:
        all_adrs += [all_adrs[0]]
        cond = [
            f"{all_adrs[index]} = {all_adrs[index + 1]}"
            for index in range(len(all_adrs) - 1)
        ]
        where_conds.append(" AND ".join(cond))

    def _construct_id_list(self, ids: List[str]) -> str:
        return ", ".join([f"'{id_}'" for id_ in ids])

    def _construct_where_anchor_node(
        self, node: QueryGraphNode, where_conds: List[str]
    ) -> None:
        node_addr = self._get_all_sql_address_for_var(node.id_)[0]
        qids = self.graph.matched_anchors_qids[node.id_]
        qids_list = self._construct_id_list(qids)
        where_conds.append(f"{node_addr} IN ({qids_list})")

    def _construct_where(self) -> str:
        # NOTE We cut out some logic related to templated QueryGraphs.
        where_conds: List[str] = []
        for index, (_, edges) in enumerate(self.edge_list):
            # TODO(jlscheerer) Update this assertion, mimic legacy behavior.
            assert len(edges) == 1
            edge = edges[0]

            where_conds.append(
                f"c{index}.property IN ({self._construct_id_list(edge.get_matched_pids())})"
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

    def _strip_id_if_needed(self, column: ColumnInfo) -> str:
        column_name = f"{column}"
        if column_name.endswith(" (pid)") or column_name.endswith(" (qid)"):
            return column_name[: -len(" (pid)")]
        return column_name

    def _join_query_labels(self, query: str, columns: List[ColumnInfo]) -> str:
        selections = ", ".join(
            [
                f'oq."{column}", l{index}.value AS "{self._strip_id_if_needed(column)} (label)"'
                for (index, column) in enumerate(columns)
            ]
        )
        label_joins = " ".join(
            [
                f'LEFT JOIN labels_en l{index} ON l{index}.id = oq."{column}"'
                for (index, column) in enumerate(columns)
            ]
        )
        return f"""WITH orig_query AS ({query})
                   SELECT {selections}
                   FROM orig_query oq {label_joins};"""


def wqg2sql(
    wqg: ExecutableQueryGraph, stats: QueryStatistics, emit_labels: bool = False
) -> Tuple[str, QueryStatistics]:
    sql = SQLBackend(wqg).to_query(stats, emit_labels)
    return sql, stats
