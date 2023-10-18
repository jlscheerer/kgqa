from enum import Enum, auto
from typing import Generic, List, Tuple, TypeVar, Union, Optional
from dataclasses import dataclass

from rdflib.plugins import sparql
from rdflib import term

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
        return BinaryExpression(lhs, op, rhs)
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
