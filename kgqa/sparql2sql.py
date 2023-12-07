from collections import defaultdict
from enum import Enum, auto
import inspect
from typing import Generic, List, Set, Tuple, TypeVar, Union, Optional
from dataclasses import dataclass
import re
import typing

from rdflib.plugins import sparql
from rdflib import term

from .QueryBackend import QueryString
from .SPARQLBackend import PropertyOccurrence, SPARQLQuery

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
    localname: Optional[str]

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
    "BuiltinExpression",
    List["Expression"],
]


class BinaryExpressionType(Enum):
    IN = auto()
    EQ = auto()


class BuiltinExpressionType(Enum):
    STRAFTER = auto()
    STR = auto()


@dataclass
class BinaryExpression:
    lhs: Expression
    op: BinaryExpressionType
    rhs: Expression


@dataclass
class BuiltinExpression:
    type_: BuiltinExpressionType
    args: List[Expression]


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
    return PName(
        prefix=pname["prefix"],
        localname=pname["localname"] if "localname" in pname else None,
    )


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
    if isinstance(expr, list):
        # We have a list of subexpressions
        return [_parse_sparql_expr(subexpr) for subexpr in expr]
    elif isinstance(expr, term.Variable):
        return _parse_sparql_variable(expr)
    elif isinstance(expr, term.Literal):
        return _parse_sparql_literal(expr)
    elif expr.name == "Builtin_STRAFTER":
        assert "arg1" in expr and "arg2" in expr
        arg1 = _parse_sparql_expr(expr["arg1"])
        arg2 = _parse_sparql_expr(expr["arg2"])

        return BuiltinExpression(BuiltinExpressionType.STRAFTER, [arg1, arg2])
    elif expr.name == "Builtin_STR":
        assert "arg" in expr
        arg = _parse_sparql_expr(expr["arg"])
        return BuiltinExpression(BuiltinExpressionType.STR, [arg])
    elif expr.name in EXPR_AGGREGATE_TYPES:
        distinct = bool(expr["distinct"])
        var = _parse_sparql_expr(expr["vars"])
        assert isinstance(var, Variable)
        return AggregateExpression(
            type_=expr.name[len("Aggregate_") :].upper(), distinct=distinct, var=var
        )
    elif expr.name == "pname":
        return _parse_sparql_pname(expr)
    elif set(expr.keys()) == {"prefix", "localname"}:
        return _parse_sparql_pname(expr)
    elif set(expr.keys()) == {"expr"}:
        return _parse_sparql_expr(expr["expr"])
    elif set(expr.keys()) == {"expr", "op", "other"}:
        lhs = _parse_sparql_expr(expr["expr"])
        op = expr["op"]
        if op == "=":
            op = "EQ"
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


class VariableDescriptor(Enum):
    WD = auto()
    WDT = auto()
    P = auto()
    PS = auto()
    PQ = auto()


@dataclass
class SQLTriple(Triple):
    qualifier: Optional[Variable] = None


class SQLTranspiler:
    query: ParsedSPARQLQuery
    base_table: str

    def __init__(self, query: ParsedSPARQLQuery, base_table: str, qualifier_table: str):
        self.query = query
        self.base_table = base_table
        self.qualifier_table = qualifier_table
        self._var_types: dict[str, PropertyOccurrence] = dict()
        self._var_refs: dict[str, str] = dict()
        self.triples: List[SQLTriple] = []

    def to_query(self) -> str:
        WHERE = self._emit_sql_where()
        FROM = self._emit_sql_from()
        SELECT = self._emit_sql_select()
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
        if self.triples is not None:
            results: List[str] = []
            for index, triple in enumerate(self.triples):
                if self._is_qualifier_triple(triple):
                    results.append(f"{self.qualifier_table} q{index}")
                else:
                    results.append(f"{self.base_table} c{index}")
            return ", ".join(results)
        raise AssertionError("trying to convert SPARQL query without triples to SQL.")

    def _emit_sql_where(self) -> str:
        where_conds: List[str] = []

        if self.query.filters is not None:
            for filter in self.query.filters:
                if (
                    isinstance(filter, BinaryExpression)
                    and filter.op == BinaryExpressionType.IN
                ):
                    if not isinstance(filter.lhs, Variable):
                        raise AssertionError(
                            f"unsupported left-hand side in 'IN' expression: {filter.lhs}"
                        )
                    if not isinstance(filter.rhs, list):
                        raise AssertionError(
                            f"unsupported right-hand side in 'IN' expression: {filter.rhs}"
                        )

                    if not isinstance(filter.rhs[0], PName):
                        raise AssertionError
                    prefix = filter.rhs[0].prefix
                    if prefix not in ["wdt", "wd", "p", "ps", "pq"]:
                        raise AssertionError
                    for element in filter.rhs:
                        if not isinstance(element, PName) or element.prefix != prefix:
                            raise AssertionError(
                                f"unsupported expression in 'IN' expression: {element}"
                            )
                    self._var_types[
                        filter.lhs.name
                    ] = VariableDescriptor[  # type:ignore
                        prefix.upper()
                    ]
                elif (
                    isinstance(filter, BinaryExpression)
                    and filter.op == BinaryExpressionType.EQ
                ):
                    lhs_var, lhs_type = self._parse_str_after(filter.lhs)
                    rhs_var, rhs_type = self._parse_str_after(filter.rhs)
                    assert (
                        self._var_types[rhs_var.name]
                        == VariableDescriptor[rhs_type.upper()]
                    )
                    self._var_types[lhs_var.name] = VariableDescriptor[lhs_type.upper()]
                    self._var_refs[lhs_var.name] = rhs_var.name

        p: dict[str, Triple] = dict()
        ps: dict[str, Triple] = dict()

        # condense the triples
        if self.query.triples is not None:
            for triple in self.query.triples:
                if isinstance(triple.pred, PName):
                    self.triples.append(SQLTriple(triple.subj, triple.pred, triple.obj))
                    continue

                assert isinstance(triple.pred, Variable)
                var_type = self._var_types[triple.pred.name]
                if var_type == VariableDescriptor.P:
                    assert isinstance(triple.obj, Variable)
                    assert triple.obj.name not in p
                    p[triple.obj.name] = triple
                    continue
                elif var_type == VariableDescriptor.PS:
                    assert isinstance(triple.subj, Variable)
                    assert triple.subj.name not in ps
                    ps[triple.subj.name] = triple
                    continue

                self.triples.append(SQLTriple(triple.subj, triple.pred, triple.obj))
        assert p.keys() == ps.keys()
        for claim in p.keys():
            p_claim = p[claim]
            ps_claim = ps[claim]
            self.triples.append(
                SQLTriple(
                    p_claim.subj, p_claim.pred, ps_claim.obj, qualifier=ps_claim.subj  # type: ignore
                )
            )

        # add the filter conditions
        if self.query.filters is not None:
            for filter in self.query.filters:
                # NOTE we only support the Literal "True" and the BinaryExpression "IN"
                if isinstance(filter, Literal):
                    assert isinstance(filter.value, bool)
                    assert filter.value
                    where_conds.append("true")
                elif (
                    isinstance(filter, BinaryExpression)
                    and filter.op == BinaryExpressionType.IN
                ):
                    if not isinstance(filter.lhs, Variable):
                        raise AssertionError(
                            f"unsupported left-hand side in 'IN' expression: {filter.lhs}"
                        )
                    if not isinstance(filter.rhs, list):
                        raise AssertionError(
                            f"unsupported right-hand side in 'IN' expression: {filter.rhs}"
                        )

                    if not isinstance(filter.rhs[0], PName):
                        raise AssertionError
                    prefix = filter.rhs[0].prefix
                    id_list = ", ".join(
                        [f"'{element.localname}'" for element in filter.rhs]  # type: ignore
                    )
                    ref = self._find_references(filter.lhs)[0]
                    where_conds.append(f"{ref} IN ({id_list})")
                elif (
                    isinstance(filter, BinaryExpression)
                    and filter.op == BinaryExpressionType.EQ
                ):
                    continue
                else:
                    raise AssertionError(
                        f"unsupported filter in SPARQL query: {filter}"
                    )

        # add the triple condition constraints
        for index, triple in enumerate(self.triples):
            if not isinstance(triple.subj, Variable):
                assert isinstance(triple.subj, PName)
                where_conds.append(
                    self._emit_eq_literal_constraint(
                        f"c{index}.entity_id", triple.subj.localname  # type: ignore
                    )
                )
            if not isinstance(triple.pred, Variable):
                assert isinstance(triple.pred, PName)
                where_conds.append(
                    self._emit_eq_literal_constraint(
                        f"c{index}.property", triple.pred.localname  # type: ignore
                    )
                )
            if not isinstance(triple.obj, Variable):
                assert isinstance(triple.obj, PName)
                where_conds.append(
                    self._emit_eq_literal_constraint(
                        f"c{index}.datavalue_entity", triple.obj.localname  # type: ignore
                    )
                )

        for var in self._collect_all_vars():
            refs = self._find_references(var)
            constr = self._emit_eq_constraint(refs)
            if constr == "True":
                continue
            where_conds.append(constr)
        assert len(where_conds) > 0
        return " AND ".join(where_conds)

    def _parse_str_after(self, expr: Expression) -> Tuple[Variable, str]:
        assert (
            isinstance(expr, BuiltinExpression)
            and expr.type_ == BuiltinExpressionType.STRAFTER
        )
        assert (
            isinstance(expr.args[0], BuiltinExpression)
            and expr.args[0].type_ == BuiltinExpressionType.STR
        )
        variable = expr.args[0].args[0]
        assert isinstance(variable, Variable)

        assert (isinstance(expr.args[1], BuiltinExpression)) and expr.args[
            1
        ].type_ == BuiltinExpressionType.STR
        pname = expr.args[1].args[0]
        assert isinstance(pname, PName) and pname.localname is None

        return variable, pname.prefix

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
        elif type_ == "qualifier":
            return "id"
        raise AssertionError

    def _emit_sql_group_by(self, group_by: List[Variable]):
        return ", ".join([f"{self._find_references(var)[0]}" for var in group_by])

    def _emit_eq_literal_constraint(self, field: str, literal: str):
        return f"{field} = '{literal}'"

    def _emit_eq_constraint(self, refs: List[str]) -> str:
        refs += [refs[0]]
        cond = [
            f"{refs[index]} = {refs[index + 1]}"
            for index in range(len(refs) - 1)
            if refs[index] != refs[index + 1]
        ]
        if len(cond) == 0:
            return "True"
        return " AND ".join(cond)

    def _collect_all_vars(self) -> Set[Variable]:
        vars: Set[Variable] = set()
        for element in self.query.projection:
            if isinstance(element, Variable):
                vars.add(element)
            elif isinstance(element, NamedExpression):
                assert isinstance(element.expression, AggregateExpression)
                vars.add(element.expression.var)

        for triple in self.triples:
            if isinstance(triple.subj, Variable):
                vars.add(triple.subj)
            if isinstance(triple.pred, Variable):
                vars.add(triple.pred)
            if isinstance(triple.obj, Variable):
                vars.add(triple.obj)
            if triple.qualifier is not None:
                vars.add(triple.qualifier)

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
        for index, triple in enumerate(self.triples):
            prefix = "q" if self._is_qualifier_triple(triple) else "c"
            if var == triple.subj:
                if self._sql_type_for_variable(var) == "datavalue_entity":
                    refs.append(f"{prefix}{index}.entity_id")
                elif self._sql_type_for_variable(
                    var
                ) == "id" and self._is_qualifier_triple(triple):
                    refs.append(f"q{index}.claim_id")
                elif self._sql_type_for_variable(var) == "id":
                    refs.append(f"c{index}.id")
                else:
                    assert False
            elif var == triple.pred:
                if self._is_qualifier_triple(triple):
                    refs.append(f"{prefix}{index}.qualifier_property")
                else:
                    refs.append(f"{prefix}{index}.property")
            elif var == triple.obj:
                refs.append(f"{prefix}{index}.{self._sql_type_for_variable(var)}")
            elif var == triple.qualifier:
                refs.append(f"c{index}.id")
        if len(refs) == 0:
            raise AssertionError(f"could not find reference for variable: {var}")
        return refs

    def _is_qualifier_triple(self, triple: SQLTriple):
        if not isinstance(triple.pred, Variable):
            return False
        return self._var_types[triple.pred.name] == VariableDescriptor.PQ


def sparql2sql(
    query: SPARQLQuery,
    base_table="claims_inv",
    qualifier_table="qualifiers",
    assert_wiki=True,
) -> SQLQuery:
    return SQLQuery(
        value=SQLTranspiler(
            parse_sparql_query(query.value, assert_wiki=assert_wiki),
            base_table=base_table,
            qualifier_table=qualifier_table,
        ).to_query()
    )
