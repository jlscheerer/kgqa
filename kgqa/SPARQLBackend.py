from dataclasses import dataclass
from typing import Tuple
from typing_extensions import override

from kgqa.QueryBackend import QueryBackend, QueryString
from kgqa.QueryGraph import (
    AnchorEntityColumnInfo,
    ColumnInfo,
    ExecutableQueryGraph,
    HeadEntityColumnInfo,
    PropertyColumnInfo,
    QueryStatistics,
)


@dataclass
class SPARQLQuery(QueryString):
    pass


class SPARQLBackend(QueryBackend):
    # Maps columns to corresponding SPARQL names.
    col2name: dict[ColumnInfo, str] = dict()

    @override
    def to_query(
        self, stats: QueryStatistics, emit_labels: bool = False
    ) -> SPARQLQuery:
        if emit_labels:
            raise AssertionError("unsupported option 'emit_labels' for SPARQLBackend")

        SELECT = self._construct_select()
        WHERE = self._construct_where()
        FILTER = self._construct_filter()
        query = f"""SELECT {SELECT}
                    WHERE
                    {{
                        {WHERE}
                        FILTER({FILTER})
                        SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
                    }}
                 """

        stats.set_column_info(self.columns)

        return SPARQLQuery(value=query)

    def _construct_select(self) -> str:
        return " ".join(
            [self._sparql_name_for_column(column) for column in self.columns]
        )

    def _construct_where(self) -> str:
        return "?X wdt:P57 wd:Q3772 ."

    def _construct_filter(self) -> str:
        if not self.requires_filters():
            # TODO(jlscheerer) We need to check for empty here instead.
            return "True"
        return ""

    def _sparql_name_for_column(self, column: ColumnInfo) -> str:
        if column in self.col2name:
            return self.col2name[column]

        if isinstance(column, PropertyColumnInfo):
            self.col2name[column] = f"?P{len(self.col2name)}"
        elif isinstance(column, AnchorEntityColumnInfo):
            self.col2name[column] = f"?A{len(self.col2name)}"
        elif isinstance(column, HeadEntityColumnInfo):
            self.col2name[column] = f"?H{len(self.col2name)}"

        return self.col2name[column]


def wqg2sparql(
    wqg: ExecutableQueryGraph, stats: QueryStatistics, emit_labels: bool = False
) -> Tuple[SPARQLQuery, QueryStatistics]:
    sparql = SPARQLBackend(wqg).to_query(stats, emit_labels)
    return sparql, stats
