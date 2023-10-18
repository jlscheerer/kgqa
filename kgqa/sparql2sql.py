from enum import Enum, auto
import inspect
from typing import Generic, List, Set, Tuple, TypeVar, Union, Optional
from dataclasses import dataclass
from numpy import isin

from rdflib.plugins import sparql
from rdflib import term

from kgqa.SPARQLBackend import SPARQLQuery
from kgqa.SQLBackend import SQLQuery

T = TypeVar("T")


@dataclass
class Literal(Generic[T]):
    value: T

    def __repr__(self) -> str:
        return f"Literal({self.value})"


@dataclass
class Variable:
    name: str

    def __repr__(self):
        return f"?{self.name}"

    def __eq__(self, other) -> bool:
        if not isinstance(other, Variable):
            return False
        return self.name == other.name

    def __hash__(self):
        return hash(self.name)


@dataclass
class PName:
    """
    Represents a SPARQL path name (PNAME)
    """

    prefix: str
    localname: str

    def __repr__(self):
        return f"{self.prefix}:{self.localname}"


Expression = Union[Literal, Variable, PName, "BinaryExpression", List["Expression"]]


class BinaryExpressionType(Enum):
    IN = auto()


@dataclass
class BinaryExpression:
    lhs: Expression
    op: BinaryExpressionType
    rhs: Expression


TripleConstituent = Union[Variable, PName]


@dataclass
class Triple:
    subj: TripleConstituent
    pred: TripleConstituent
    obj: TripleConstituent


@dataclass
class ParsedSPARQLQuery:
    projection: List[Variable]
    triples: Optional[List[Triple]]
    filters: Optional[List[Expression]]


def _parse_sparql_variable(var: term.Variable) -> Variable:
    assert isinstance(var, term.Variable)
    name = var.toPython()
    assert name.startswith("?")
    return Variable(name=name[1:])


def _parse_sparql_literal(literal: term.Literal) -> Literal:
    assert isinstance(literal.value, bool)
    return Literal(value=literal.value)


def _parse_sparql_projection(projection) -> List[Variable]:
    vars: List[Variable] = []
    for var in [x["var"] for x in projection]:
        assert isinstance(var, term.Variable)
        vars.append(_parse_sparql_variable(var))
    return vars


def _parse_sparql_pname(pname) -> PName:
    return PName(prefix=pname["prefix"], localname=pname["localname"])


def _parse_triple_constituent(constituent) -> TripleConstituent:
    # NOTE an element of a triple is either a variable, a pname or a PathAlternative
    if isinstance(constituent, term.Variable):
        return _parse_sparql_variable(constituent)
    elif isinstance(constituent, list):
        assert len(constituent) == 1
        return _parse_triple_constituent(constituent[0])
    elif set(constituent.keys()) == {"prefix", "localname"}:
        return _parse_sparql_pname(constituent)

    assert set(constituent.keys()) == {"part"}
    return _parse_triple_constituent(constituent["part"])


def _parse_sparql_triple(triple) -> Triple:
    assert len(triple) == 3
    constituents = [_parse_triple_constituent(constituent) for constituent in triple]
    return Triple(subj=constituents[0], pred=constituents[1], obj=constituents[2])


def _parse_sparql_triples(triples) -> List[Triple]:
    return [_parse_sparql_triple(triple) for triple in triples]


def _parse_sparql_expr(expr) -> Expression:
    if isinstance(expr, list):
        # We have a list of subexpressions
        return [_parse_sparql_expr(subexpr) for subexpr in expr]
    elif isinstance(expr, term.Variable):
        return _parse_sparql_variable(expr)
    elif isinstance(expr, term.Literal):
        return _parse_sparql_literal(expr)
    elif set(expr.keys()) == {"prefix", "localname"}:
        return _parse_sparql_pname(expr)
    elif set(expr.keys()) == {"expr"}:
        return _parse_sparql_expr(expr["expr"])
    elif set(expr.keys()) == {"expr", "op", "other"}:
        lhs = _parse_sparql_expr(expr["expr"])
        op = expr["op"]
        rhs = _parse_sparql_expr(expr["other"])
        return BinaryExpression(lhs, BinaryExpressionType[op], rhs)
    # We have a list of subexpressions
    return [_parse_sparql_expr(subexpr) for subexpr in expr]


def _parse_sparql_filter(filter) -> List[Expression]:
    assert filter.name == "Filter"
    expr = filter["expr"]["expr"]

    assert set(expr.keys()) == {"expr"} or (expr.keys() == {"expr", "other"})

    filters = [_parse_sparql_expr(expr["expr"])]
    if "other" in expr.keys():
        tail = _parse_sparql_expr(expr["other"])
        assert isinstance(tail, list)
        filters.extend(tail)

    return filters


def _parse_sparql_where(
    where,
) -> Tuple[Optional[List[Triple]], Optional[List[Expression]]]:
    triples = None
    filter = None
    is_wiki_service = False

    for constituent in where["part"]:
        if "triples" in constituent:
            assert constituent.name == "TriplesBlock" and triples is None
            triples = _parse_sparql_triples(constituent["triples"])
        elif "expr" in constituent:
            assert constituent.name == "Filter" and filter is None
            filter = _parse_sparql_filter(constituent)
        elif "service_string" in constituent:
            assert constituent.name == "ServiceGraphPattern" and not is_wiki_service
            assert "term" in constituent
            assert "graph" in constituent

            is_wiki_service = True

    assert is_wiki_service
    return triples, filter


def parse_sparql_query(query: str) -> ParsedSPARQLQuery:
    parsed = sparql.parser.parseQuery(query)[1]
    projection = _parse_sparql_projection(parsed["projection"])
    triples, filter = _parse_sparql_where(parsed["where"])

    return ParsedSPARQLQuery(projection, triples, filter)


class SQLTranspiler:
    query: ParsedSPARQLQuery
    base_table: str

    def __init__(self, query: ParsedSPARQLQuery, base_table: str = "claims_5m_inv"):
        self.query = query
        self.base_table = base_table

    def to_query(self) -> str:
        SELECT = self._emit_sql_select()
        FROM = self._emit_sql_from()
        WHERE = self._emit_sql_where()

        return inspect.cleandoc(
            f"""SELECT {SELECT}
                FROM {FROM}
                WHERE {WHERE};"""
        )

    def _emit_sql_select(self) -> str:
        projections: List[str] = []
        for column in self.query.projection:
            ref = self._find_references(column)[0]
            projections.append(f'{ref} AS "{column}"')
        return ", ".join(projections)

    def _emit_sql_from(self) -> str:
        if self.query.triples is not None:
            return ", ".join(
                [
                    f"{self.base_table} c{index}"
                    for index in range(len(self.query.triples))
                ]
            )
        raise AssertionError("trying to convert SPARQL query without triples to SQL.")

    def _emit_sql_where(self) -> str:
        where_conds: List[str] = []
        # TODO(jlscheerer) add the triple conditions...
        for var in self._collect_all_vars():
            refs = self._find_references(var)
            where_conds.append(self._emit_eq_constraint(refs))

        # add the filter conditions
        if self.query.filters is not None:
            for filter in self.query.filters:
                # NOTE we only support the Literal "True" and the BinaryExpression "IN"
                if isinstance(filter, Literal):
                    assert isinstance(filter.value, bool)
                    assert filter.value
                    where_conds.append("true")
                elif isinstance(filter, BinaryExpression):
                    if filter.op != BinaryExpressionType.IN:
                        raise AssertionError(
                            f"unsupported binary expression in SPARQL query: {filter.op}"
                        )
                    if not isinstance(filter.lhs, Variable):
                        raise AssertionError(
                            f"unsupported left-hand side in 'IN' expression: {filter.lhs}"
                        )
                    if not isinstance(filter.rhs, list):
                        raise AssertionError(
                            f"unsupported right-hand side in 'IN' expression: {filter.rhs}"
                        )
                    for element in filter.rhs:
                        if not isinstance(element, PName) or element.prefix not in (
                            "wdt",
                            "wd",
                        ):
                            raise AssertionError(
                                f"unsupported expression in 'IN' expression: {element}"
                            )
                    id_list = ", ".join(
                        [f"'{element.localname}'" for element in filter.rhs]  # type: ignore
                    )
                    ref = self._find_references(filter.lhs)[0]
                    where_conds.append(f"{ref} IN ({id_list})")
                else:
                    raise AssertionError(
                        f"unsupported filter in SPARQL query: {filter}"
                    )

        assert len(where_conds) > 0
        return " AND ".join(where_conds)

    def _emit_eq_constraint(self, refs: List[str]) -> str:
        refs += [refs[0]]
        cond = [f"{refs[index]} = {refs[index + 1]}" for index in range(len(refs) - 1)]
        return " AND ".join(cond)

    def _collect_all_vars(self) -> Set[Variable]:
        vars: Set[Variable] = set()
        for var in self.query.projection:
            vars.add(var)

        if self.query.triples is not None:
            for triple in self.query.triples:
                if isinstance(triple.subj, Variable):
                    vars.add(triple.subj)
                if isinstance(triple.pred, Variable):
                    vars.add(triple.pred)
                if isinstance(triple.obj, Variable):
                    vars.add(triple.obj)

        if self.query.filters is not None:
            for filter in self.query.filters:
                if isinstance(filter, BinaryExpression):
                    if isinstance(filter.lhs, Variable):
                        vars.add(filter.lhs)

        return vars

    def _find_references(self, var: Variable) -> List[str]:
        refs: List[str] = []
        if self.query.triples is None:
            raise AssertionError(
                f"trying to find reference for variable ' {var}' with empty triples."
            )
        for index, triple in enumerate(self.query.triples):
            if var == triple.subj:
                refs.append(f"c{index}.entity_id")
            elif var == triple.pred:
                refs.append(f"c{index}.property")
            elif var == triple.obj:
                refs.append(f"c{index}.datavalue_entity")
        if len(refs) == 0:
            raise AssertionError(f"could not find reference for variable: {var}")
        return refs


def sparql2sql(query: SPARQLQuery) -> SQLQuery:
    return SQLQuery(value=SQLTranspiler(parse_sparql_query(query.value)).to_query())
