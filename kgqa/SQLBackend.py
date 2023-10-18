from dataclasses import dataclass
from typing import Tuple, List
from typing_extensions import override

from .QueryBackend import (
    QueryBackend,
    QueryString,
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


@dataclass
class SQLQuery(QueryString):
    pass


class SQLBackend(QueryBackend):
    @override
    def to_query(self, stats: QueryStatistics, emit_labels: bool = False) -> SQLQuery:
        SELECT = self._construct_select()
        FROM = self._construct_from()
        WHERE = self._construct_where()
        query = f"""SELECT {SELECT} FROM {FROM} WHERE {WHERE}"""

        stats.set_column_info(self.columns)
        if emit_labels:
            query = self._join_query_labels(query, stats.columns)
        else:
            query = f"{query};"

        return SQLQuery(value=query)

    def _construct_from(self) -> str:
        # NOTE We cut out some logic related to templated QueryGraphs.

        claims_relname = "claims_5m_inv"
        return ", ".join(
            [f"{claims_relname} c{index}" for index in range(len(self.edge_list))]
        )

    def _get_references_for_var(self, var_id: QueryGraphId) -> List[str]:
        refs: List[str] = []

        # TODO(jlscheerer) We would need to support filters here.
        assert not self.requires_filters()

        object_type = "datavalue_entity"
        for occurrence in self.var2edges[var_id]:
            so_type = "entity_id" if occurrence.is_subj else object_type
            refs.append(f"c{occurrence.edge_index}.{so_type}")

        return refs

    def _construct_select(self) -> str:
        # NOTE We cut out some logic related to templated QueryGraphs.
        selects: List[str] = []

        for column in self.columns:
            if isinstance(column, PropertyColumnInfo):
                selects.append(f'c{column.index}.property AS "{column}"')
            elif isinstance(column, AnchorEntityColumnInfo):
                # NOTE for the purpose of the select, it does not matter what column we
                #      reference, so we just pick the first.
                node_addr = self._get_references_for_var(column.index)[0]
                selects.append(f'{node_addr} AS "{column}"')
            else:
                assert isinstance(column, HeadEntityColumnInfo)
                # NOTE for the purpose of the select, it does not matter what column we
                #      reference, so we just pick the first.
                node_addr = self._get_references_for_var(column.index)[0]
                selects.append(f'{node_addr} AS "{column}"')

        return ", ".join(selects)

    def _construct_where_join_condition(
        self, where_conds: List[str], all_adrs: List[str]
    ) -> None:
        # TODO(jlscheerer) This constructs a cyclic pattern. Is this required?
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
        node_addr = self._get_references_for_var(node.id_)[0]
        qids = self.graph.matched_anchors_qids[node.id_]
        qids_list = self._construct_id_list(qids)
        where_conds.append(f"{node_addr} IN ({qids_list})")

    def _construct_where(self) -> str:
        # NOTE We cut out some logic related to templated QueryGraphs.
        where_conds: List[str] = []
        for index, (_, _, edge) in enumerate(self.edge_list):
            where_conds.append(
                f"c{index}.property IN ({self._construct_id_list(edge.get_matched_pids())})"
            )
        for node in self.graph.nodes:
            refs = self._get_references_for_var(node.id_)
            if len(refs) >= 2:
                self._construct_where_join_condition(where_conds, refs)
            if not node.is_free:
                self._construct_where_anchor_node(node, where_conds)

        # TODO(jlscheerer) We would need to support filters here.
        assert not self.requires_filters()

        return " AND ".join(where_conds)

    def _join_query_labels(self, query: str, columns: List[ColumnInfo]) -> str:
        """
        Joins with labels_en for each claims table occuring in `query`.
        """
        selections = ", ".join(
            [
                f'ulq."{column}", l{index}.value AS "{column.base_name()} (label)"'
                for (index, column) in enumerate(columns)
            ]
        )
        label_joins = " ".join(
            [
                f'LEFT JOIN labels_en l{index} ON l{index}.id = ulq."{column}"'
                for (index, column) in enumerate(columns)
            ]
        )
        return f"""WITH unlabeled_query AS ({query})
                   SELECT {selections}
                   FROM unlabeled_query ulq {label_joins};"""


def wqg2sql(
    wqg: ExecutableQueryGraph, stats: QueryStatistics, emit_labels: bool = False
) -> Tuple[SQLQuery, QueryStatistics]:
    sql = SQLBackend(wqg).to_query(stats, emit_labels)
    return sql, stats
