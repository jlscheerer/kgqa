from enum import Enum, auto
import inspect
from typing import Generic, List, Set, Tuple, TypeVar, Union, Optional
from dataclasses import dataclass
import re
import typing

from rdflib.plugins import sparql
from rdflib import term

from .QueryBackend import QueryString
from .SPARQLBackend import SPARQLQuery

T = TypeVar("T")


@dataclass
class SQLQuery(QueryString):
    pass


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
class SPARQLVariableTypeInfo:
    variable: Variable
    type_: typing.Literal[
        "entity_id", "string", "date", "numeric", "coordinate", "qualifier"
    ]


@dataclass
class PName:
    """
    Represents a SPARQL path name (PNAME)
    """

    prefix: str
    localname: str

    def __repr__(self):
        return f"{self.prefix}:{self.localname}"


EXPR_AGGREGATE_TYPES = [
    "Aggregate_Count",
    "Aggregate_Sum",
    "Aggregate_Avg",
    "Aggregate_Min",
    "Aggregate_Max",
    "Aggregate_Sample",
    "Aggregate_GroupConcat",
]


@dataclass
class AggregateExpression:
    type_: str
    distinct: bool
    var: Variable


Expression = Union[
    Literal,
    Variable,
    PName,
    AggregateExpression,
    "BinaryExpression",
    List["Expression"],
]


class BinaryExpressionType(Enum):
    IN = auto()


@dataclass
class BinaryExpression:
    lhs: Expression
    op: BinaryExpressionType
    rhs: Expression


@dataclass
class NamedExpression:
    name: Variable
    expression: Expression


TripleConstituent = Union[Variable, PName]


@dataclass
class Triple:
    subj: TripleConstituent
    pred: TripleConstituent
    obj: TripleConstituent


@dataclass
class ParsedSPARQLQuery:
    projection: List[Union[Variable, NamedExpression]]
    triples: Optional[List[Triple]]
    filters: Optional[List[Expression]]
    group_by: Optional[List[Variable]]
    type_info: List[SPARQLVariableTypeInfo]


def _parse_sparql_variable(var: term.Variable) -> Variable:
    assert isinstance(var, term.Variable)
    name = var.toPython()
    assert name.startswith("?")
    return Variable(name=name[1:])


def _parse_sparql_literal(literal: term.Literal) -> Literal:
    assert isinstance(literal.value, bool)
    return Literal(value=literal.value)


def _parse_sparql_projection(projection) -> List[Union[Variable, NamedExpression]]:
    vars: List[Union[Variable, NamedExpression]] = []
    for element in projection:
        if "var" in element:
            var = element["var"]
            assert isinstance(var, term.Variable)
            vars.append(_parse_sparql_variable(var))
            continue
        assert "evar" in element
        vars.append(
            NamedExpression(
                name=_parse_sparql_variable(element["evar"]),
                expression=_parse_sparql_expr(element["expr"]),
            )
        )
    return vars


def _parse_sparql_pname(pname) -> PName:
    return PName(prefix=pname["prefix"], localname=pname["localname"])


def _parse_triple_constituent(constituent) -> TripleConstituent:
    # NOTE an element of a triple is either a variable, a pname or a PathAlternative
    if isinstance(constituent, term.Variable):
        return _parse_sparql_variable(constituent)
    elif isinstance(constituent, list):
        # TODO(jlscheerer) Support 9.1 Property Path Syntax
        # print(*[x for x in constituent], sep="\n")
        assert len(constituent) == 1
        return _parse_triple_constituent(constituent[0])
    elif set(constituent.keys()) == {"prefix", "localname"}:
        return _parse_sparql_pname(constituent)

    assert set(constituent.keys()) == {"part"}
    return _parse_triple_constituent(constituent["part"])


def _parse_sparql_triple(triple) -> List[Triple]:
    constituents = [_parse_triple_constituent(constituent) for constituent in triple]

    # NOTE we can have multiple triples here as part of (4.2.1 Predicate-Object Lists)
    assert len(constituents) > 0 and len(constituents) % 3 == 0
    return [
        Triple(
            subj=constituents[offset + 0],
            pred=constituents[offset + 1],
            obj=constituents[offset + 2],
        )
        for offset in range(0, len(constituents), 3)
    ]


def _parse_sparql_triples(triples) -> List[Triple]:
    ptriples: List[Triple] = []
    for triple in triples:
        ptriples.extend(_parse_sparql_triple(triple))
    return ptriples


def _parse_sparql_expr(expr) -> Expression:
    # print(expr)
    if isinstance(expr, list):
        # We have a list of subexpressions
        return [_parse_sparql_expr(subexpr) for subexpr in expr]
    elif isinstance(expr, term.Variable):
        return _parse_sparql_variable(expr)
    elif isinstance(expr, term.Literal):
        return _parse_sparql_literal(expr)
    elif expr.name in EXPR_AGGREGATE_TYPES:
        distinct = bool(expr["distinct"])
        var = _parse_sparql_expr(expr["vars"])
        assert isinstance(var, Variable)
        return AggregateExpression(
            type_=expr.name[len("Aggregate_") :].upper(), distinct=distinct, var=var
        )
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
    where, assert_wiki=True
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
    if assert_wiki:
        assert is_wiki_service
    return triples, filter


def _parse_sparql_group_by(group_by):
    assert group_by.name == "GroupClause"
    return [_parse_sparql_variable(var) for var in group_by["condition"]]


def parse_sparql_query(query: str, assert_wiki=True) -> ParsedSPARQLQuery:
    type_info = re.findall(
        "#pragma:type \\?([^\\s]+) (string|entity_id|date|numeric|coordinate|qualifier)\n",
        query,
    )
    types: List[SPARQLVariableTypeInfo] = []
    for var_name, type_ in type_info:
        types.append(
            SPARQLVariableTypeInfo(variable=Variable(name=var_name), type_=type_)
        )

    parsed = sparql.parser.parseQuery(query)[1]
    projection = _parse_sparql_projection(parsed["projection"])
    triples, filter = _parse_sparql_where(parsed["where"], assert_wiki=assert_wiki)
    group_by = (
        _parse_sparql_group_by(parsed["groupby"]) if "groupby" in parsed else None
    )

    return ParsedSPARQLQuery(projection, triples, filter, group_by, type_info=types)


class SQLTranspiler:
    query: ParsedSPARQLQuery
    base_table: str

    def __init__(self, query: ParsedSPARQLQuery, base_table: str):
        self.query = query
        self.base_table = base_table

    def to_query(self) -> str:
        SELECT = self._emit_sql_select()
        FROM = self._emit_sql_from()
        WHERE = self._emit_sql_where()
        GROUP_BY = None
        if self.query.group_by is not None:
            GROUP_BY = f"GROUP BY {self._emit_sql_group_by(self.query.group_by)}"
            return inspect.cleandoc(
                f"""SELECT {SELECT}
                    FROM {FROM}
                    WHERE {WHERE}
                    {GROUP_BY};"""
            )

        return inspect.cleandoc(
            f"""SELECT {SELECT}
                FROM {FROM}
                WHERE {WHERE};"""
        )

    def _emit_sql_select(self) -> str:
        projections: List[str] = []
        for column in self.query.projection:
            if isinstance(column, Variable):
                ref = self._find_references(column)[0]
                projections.append(f'{ref} AS "{column}"')
            elif isinstance(column, NamedExpression):
                assert isinstance(column.expression, AggregateExpression)
                projections.append(
                    f'{self._emit_aggregate(column.expression)} AS "{column.name}"'
                )
            else:
                assert False
        return ", ".join(projections)

    def _emit_aggregate(self, aggregate: AggregateExpression) -> str:
        distinct = "DISTINCT " if aggregate.distinct else ""
        ref = self._find_references(aggregate.var)[0]
        if aggregate.type_ in ["COUNT", "MIN", "MAX", "SUM"]:
            return f"{aggregate.type_}({distinct}{ref})"
        raise AssertionError(f"unsupported aggregate expression '{aggregate.type_}'")

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

        # add the triple condition constraints
        if self.query.triples is not None:
            for index, triple in enumerate(self.query.triples):
                if not isinstance(triple.subj, Variable):
                    assert isinstance(triple.subj, PName)
                    where_conds.append(
                        self._emit_eq_literal_constraint(
                            f"c{index}.entity_id", triple.subj.localname
                        )
                    )
                if not isinstance(triple.pred, Variable):
                    assert isinstance(triple.pred, PName)
                    where_conds.append(
                        self._emit_eq_literal_constraint(
                            f"c{index}.property", triple.pred.localname
                        )
                    )
                if not isinstance(triple.obj, Variable):
                    assert isinstance(triple.obj, PName)
                    where_conds.append(
                        self._emit_eq_literal_constraint(
                            f"c{index}.datavalue_entity", triple.obj.localname
                        )
                    )

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

    def _sql_type_for_variable(self, variable: Variable):
        if len(self.query.type_info) == 0:
            return "entity_id"
        for type_ in self.query.type_info:
            if type_.variable == variable:
                return self._sql_type(type_.type_)
        raise AssertionError

    def _sql_type(self, type_):
        # Only applies to object; not the subject!
        if type_ == "entity_id":
            return "datavalue_entity"
        elif type_ == "string":
            return "datavalue_string"
        elif type_ == "date":
            return "datavalue_date"
        elif type_ == "numeric":
            return "datavalue_quantity"
        raise AssertionError

    def _emit_sql_group_by(self, group_by: List[Variable]):
        return ", ".join([f"{self._find_references(var)[0]}" for var in group_by])

    def _emit_eq_literal_constraint(self, field: str, literal: str):
        return f"{field} = '{literal}'"

    def _emit_eq_constraint(self, refs: List[str]) -> str:
        refs += [refs[0]]
        cond = [f"{refs[index]} = {refs[index + 1]}" for index in range(len(refs) - 1)]
        return " AND ".join(cond)

    def _collect_all_vars(self) -> Set[Variable]:
        vars: Set[Variable] = set()
        for element in self.query.projection:
            if isinstance(element, Variable):
                vars.add(element)
            elif isinstance(element, NamedExpression):
                assert isinstance(element.expression, AggregateExpression)
                vars.add(element.expression.var)

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
                assert self._sql_type_for_variable(var) == "datavalue_entity"
                refs.append(f"c{index}.entity_id")
            elif var == triple.pred:
                refs.append(f"c{index}.property")
            elif var == triple.obj:
                refs.append(f"c{index}.{self._sql_type_for_variable(var)}")
        if len(refs) == 0:
            raise AssertionError(f"could not find reference for variable: {var}")
        return refs


def sparql2sql(
    query: SPARQLQuery, base_table="claims_inv", assert_wiki=True
) -> SQLQuery:
    return SQLQuery(
        value=SQLTranspiler(
            parse_sparql_query(query.value, assert_wiki=assert_wiki),
            base_table=base_table,
        ).to_query()
    )
