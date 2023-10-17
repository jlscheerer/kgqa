from dataclasses import dataclass
from typing import Tuple
from typing_extensions import override

from kgqa.QueryBackend import QueryBackend, QueryString
from kgqa.QueryGraph import ExecutableQueryGraph, QueryStatistics


@dataclass
class SPARQLQuery(QueryString):
    pass


class SPARQLBackend(QueryBackend):
    @override
    def to_query(
        self, stats: QueryStatistics, emit_labels: bool = False
    ) -> SPARQLQuery:
        raise AssertionError


def wqg2sparql(
    wqg: ExecutableQueryGraph, stats: QueryStatistics, emit_labels: bool = False
) -> Tuple[SPARQLQuery, QueryStatistics]:
    sparql = SPARQLBackend(wqg).to_query(stats, emit_labels)
    return sparql, stats
