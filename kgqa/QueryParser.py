from enum import Enum, auto
from typing import List, Optional, Set, Tuple, Union
from dataclasses import dataclass, field
import re

from .QueryLexer import (
    AggregationFunction,
    Identifier,
    LiteralToken,
    NumericLiteral,
    QueryLexer,
    QueryLexerException,
    QuoteType,
    SourceLocation,
    StringLiteral,
    Token,
    TokenType,
)


@dataclass
class Variable:
    token: Identifier
    name: str

    def source_name(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return f"Variable({self.name})"

    def __eq__(self, other) -> bool:
        if not isinstance(other, Variable):
            return False
        return self.name == other.name

    def __hash__(self):
        return hash(self.name)


@dataclass
class Constant:
    token: Union[LiteralToken, Identifier]


@dataclass
class StringConstant(Constant):
    token: StringLiteral
    value: str

    def __repr__(self) -> str:
        return f"Constant('{self.value}')"

    def __eq__(self, other) -> bool:
        if not isinstance(other, StringConstant):
            return False
        return self.value == other.value

    def __hash__(self):
        return hash(self.value)


@dataclass
class NumericConstant(Constant):
    token: NumericLiteral
    value: Union[int, float]

    def __repr__(self) -> str:
        return f"Constant({self.value})"

    def __eq__(self, other) -> bool:
        if not isinstance(other, NumericConstant):
            return False
        return self.value == other.value

    def __hash__(self):
        return hash(self.value)


class AnnotationType(Enum):
    BANG = "!"
    OPT = "?"


@dataclass
class IDConstant(Constant):
    token: Identifier
    annotation: AnnotationType
    value: str

    def source_name(self) -> str:
        return f"{self.annotation.value}{self.value}"

    def __repr__(self) -> str:
        return f"{self.annotation.value}IDConstant({self.value})"


class AggregationType(Enum):
    COUNT = auto()
    SUM = auto()
    AVG = auto()
    MAX = auto()
    MIN = auto()


@dataclass
class Aggregation:
    token: AggregationFunction
    var: Variable
    type_: AggregationType

    def __repr__(self) -> str:
        return f"{self.type_.name}({self.var})"


HeadItem = Union[Variable, Aggregation]


@dataclass
class QueryHead:
    items: List[HeadItem] = field(default_factory=lambda: [])

    def vars(self) -> List[Variable]:
        """
        Returns a list of unique variables occuring in the head, ordered by occurence.
        """
        vars: List[Variable] = []
        for item in self.items:
            if isinstance(item, Variable):
                if item not in vars:
                    vars.append(item)
            else:
                assert isinstance(item, Aggregation)
                if item.var not in vars:
                    vars.append(item.var)
        return vars

    def __iter__(self):
        return self.items.__iter__()


@dataclass
class QueryAtom:
    pass


PredicateType = Union[IDConstant, Variable]
ArgumentType = Union[Variable, Constant]


@dataclass
class QueryClause(QueryAtom):
    predicate: PredicateType
    arguments: List[ArgumentType]

    def __repr__(self) -> str:
        if isinstance(self.predicate, Variable):
            return f"{self.predicate.name}({', '.join([str(arg) for arg in self.arguments])})"
        else:
            assert isinstance(self.predicate, IDConstant)
            return (
                f"{self.predicate}({', '.join([str(arg) for arg in self.arguments])}))"
            )


class FilterOp(Enum):
    EQ = "="
    NEQ = "!="
    LE = "<"
    LEQ = "<="
    GE = ">"
    GEQ = ">="


@dataclass
class QueryFilter(QueryAtom):
    lhs: Variable
    op: FilterOp
    rhs: Constant

    def __repr__(self) -> str:
        return f"{self.lhs} {self.op.value} {self.rhs}"


@dataclass
class ParsedQuery:
    head: QueryHead

    clauses: List[QueryClause]
    filters: List[QueryFilter]

    def spo(self) -> List[Tuple[ArgumentType, PredicateType, ArgumentType]]:
        spo: List[Tuple[ArgumentType, PredicateType, ArgumentType]] = []
        for clause in self.clauses:
            if len(clause.arguments) != 2:
                raise AssertionError()
            spo.append((clause.arguments[0], clause.predicate, clause.arguments[1]))
        return spo


class QueryParserException(Exception):
    def __init__(self, token: Token, error: str):
        self.token = token
        self.error = error


class QueryParser:
    _tokens: List[Token]
    _cursor: int
    _eof: int

    def parse(self, query: str) -> ParsedQuery:
        self._tokens = list(QueryLexer(query))
        self._cursor = 0
        self._eof = len(query)
        head = QueryHead()
        if self._check_contains_colon():
            head = self._parse_head()
        clauses, filters = self._parse_body()
        pq = ParsedQuery(head=head, clauses=clauses, filters=filters)
        if not self._validate_query(pq):
            raise AssertionError()
        return self._rewrite_query(pq)

    def _rewrite_query(self, pq: ParsedQuery) -> ParsedQuery:
        # Rewrite unary queries using "instance_of": person(X) becomes !P31(X, "person")
        clauses: List[QueryClause] = []
        for clause in pq.clauses:
            if len(clause.arguments) != 1:
                clauses.append(clause)
                continue

            type_ = clause.predicate
            if not isinstance(type_, Variable):
                raise QueryParserException(
                    clause.predicate.token, "cannot rewrite unary predicate"
                )
            assert isinstance(type_, Variable)

            type_lit = StringLiteral(
                source_location=clause.predicate.token.source_location,
                value=type_.name,
            )
            clauses.append(
                QueryClause(
                    predicate=IDConstant(
                        Identifier(
                            source_location=SourceLocation.synthetic(), name="P31"
                        ),
                        annotation=AnnotationType.BANG,
                        value="P31",
                    ),
                    arguments=[
                        clause.arguments[0],
                        StringConstant(token=type_lit, value=type_lit.value),
                    ],
                )
            )
        return ParsedQuery(head=pq.head, clauses=clauses, filters=pq.filters)

    def _validate_query(self, pq: ParsedQuery) -> bool:
        head_vars = pq.head.vars()

        clause_vars: Set[Variable] = set()
        for clause in pq.clauses:
            for argument in clause.arguments:
                if isinstance(argument, Variable):
                    clause_vars.add(argument)

        # Every variable occuring in the head must be bound in the body via a clause.
        for var in head_vars:
            if var not in clause_vars:
                raise QueryParserException(
                    var.token,
                    f"unbound variable: {var.source_name()} does not appear in a clause",
                )

        # Every variable as a filter must be bound in the body via a clause.
        for filter in pq.filters:
            if filter.lhs not in clause_vars:
                raise QueryParserException(
                    filter.lhs.token,
                    f"unbound variable: {filter.lhs.source_name()} does not appear in a clause",
                )

        # Aggregations are not yet supported.
        for item in pq.head:
            if isinstance(item, Aggregation):
                raise QueryParserException(
                    item.token,
                    f"aggregation {item.type_.name}(_) not yet supported",
                )

        # We only support unary/binary predicates
        for clause in pq.clauses:
            if len(clause.arguments) == 0:
                raise QueryParserException(
                    clause.predicate.token, "unsupported nullary predicate"
                )
            elif len(clause.arguments) > 2:
                raise QueryLexerException(
                    SourceLocation(
                        begin=clause.arguments[0].token.source_location.begin,
                        end=clause.arguments[-1].token.source_location.end,
                    ),
                    f"too many arguments to predicate '{clause.predicate.source_name()}'",
                )

        # Check the format of IDConstants as entities/predicates
        for clause in pq.clauses:
            if isinstance(clause.predicate, IDConstant):
                if clause.predicate.annotation != AnnotationType.BANG:
                    raise QueryParserException(
                        clause.predicate.token,
                        f"unsupported annotation '{clause.predicate.annotation.value}' on predicate {clause.predicate.value}",
                    )
                if not re.match("^P[0-9]+$", clause.predicate.value):
                    raise QueryParserException(
                        clause.predicate.token,
                        f"unsupported format for predicate constant: {clause.predicate.value}",
                    )
            for argument in clause.arguments:
                if isinstance(argument, IDConstant):
                    if argument.annotation != AnnotationType.BANG:
                        raise QueryParserException(
                            argument.token,
                            f"unsupported annotation '{argument.annotation.value}' on predicate {argument.value}",
                        )
                    if not re.match("^Q[0-9]+", argument.value):
                        raise QueryParserException(
                            argument.token,
                            f"unsupported format for entity constant: {argument.value}",
                        )

        # Check for unsupported quote types - currently only "" is supported.
        for clause in pq.clauses:
            for argument in clause.arguments:
                if isinstance(argument, StringConstant):
                    if argument.token.quote_type != QuoteType.DOUBLE_QUOTE:
                        raise QueryParserException(
                            argument.token, "unsupported quote type for argument"
                        )

        # TODO(jlscheerer) NumericLiterals are not supported yet.

        return True

    def _parse_head(self) -> QueryHead:
        token = self._pop_token()
        # We should always at least encounter the `COLON` token.
        assert token is not None
        item: HeadItem
        if token.token_type == TokenType.IDENTIFIER:
            item = Variable(token=token, name=token.name)
        elif token.token_type == TokenType.AGGREGATION_FUNCTION:
            arguments = self._parse_arguments()
            if len(arguments) == 0:
                raise QueryParserException(token, "missing argument for aggregation")
            elif len(arguments) != 1:
                raise QueryLexerException(
                    SourceLocation(
                        begin=arguments[0].token.source_location.begin,
                        end=arguments[-1].token.source_location.end,
                    ),
                    "too many arguments for aggregation",
                )

            var = arguments[0]
            if not isinstance(var, Variable):
                raise QueryParserException(
                    var.token, f"unexpected {var} for aggregation"
                )
            item = Aggregation(token=token, var=var, type_=AggregationType[token.name])
        else:
            raise QueryParserException(
                token, f"unexpected {token.token_type} in query head"
            )

        token = self._pop_token()
        # We should always at least encounter the `COLON` token.
        assert token is not None
        if token.token_type == TokenType.COMMA:
            return QueryHead(items=[item] + self._parse_head().items)
        elif token.token_type != TokenType.COLON:
            raise QueryParserException(
                token, f"unexpected {token.token_type} in query head"
            )
        return QueryHead(items=[item])

    def _parse_body(self) -> Tuple[List[QueryClause], List[QueryFilter]]:
        clauses: List[QueryClause] = []
        filters: List[QueryFilter] = []
        while True:
            atom = self._parse_atom()
            if atom is None:
                raise QueryLexerException(
                    SourceLocation(begin=self._eof, end=self._eof + 1),
                    "missing query body",
                )

            if isinstance(atom, QueryClause):
                clauses.append(atom)
            else:
                assert isinstance(atom, QueryFilter)
                filters.append(atom)

            if self._curr_token() is None:
                break
            token = self._pop_require_token()
            if token.token_type != TokenType.SEMICOLON:
                raise QueryParserException(
                    token, f"unexpected {token.token_type} in query body"
                )
        return clauses, filters

    def _parse_atom(self) -> Optional[QueryAtom]:
        ident = self._pop_token()
        if ident is None:
            return None

        # We have an annotated predicate, e.g., !P17 / ?P17
        annotation: Optional[Token] = None
        if ident.token_type in [TokenType.EXCLAMATION_MARK, TokenType.QUESTION_MARK]:
            annotation = ident
            ident = self._pop_require_token()

        if ident.token_type != TokenType.IDENTIFIER:
            raise QueryParserException(
                ident, f"unexpected {ident.token_type} in query body"
            )
        token = self._require_token()
        if token.token_type == TokenType.OPEN_PAREN:
            arguments = self._parse_arguments()
            predicate: Union[Variable, IDConstant]
            if annotation is None:
                predicate = Variable(token=ident, name=ident.name)
            else:
                predicate = self._as_id_constant(annotation=annotation, constant=ident)
            return QueryClause(predicate=predicate, arguments=arguments)
        elif token.token_type == TokenType.COMPARATOR:
            # We cannot have annotated predicates with comparisons.
            if annotation is not None:
                raise QueryParserException(
                    annotation, f"unexpected {annotation.token_type} in comparison"
                )

            op = self._pop_require_token()
            if op.token_type != TokenType.COMPARATOR:
                raise QueryParserException(
                    op, f"expected {TokenType.COMPARATOR} got {op.token_type}"
                )
            rhs = self._pop_require_token()
            if rhs.token_type not in [
                TokenType.STRING_LITERAL,
                TokenType.NUMERIC_LITERAL,
            ]:
                raise QueryParserException(
                    rhs,
                    f"expected {TokenType.STRING_LITERAL} | {TokenType.NUMERIC_LITERAL} got {rhs.token_type}",
                )
            return QueryFilter(
                lhs=Variable(token=ident, name=ident.name),
                op=FilterOp(op.operator),
                rhs=self._as_constant(rhs),
            )
        raise QueryParserException(
            token, f"unexpected {token.token_type} in query body"
        )

    def _as_constant(self, token: Token) -> Constant:
        if token.token_type == TokenType.STRING_LITERAL:
            return StringConstant(token=token, value=token.value)
        elif token.token_type == TokenType.NUMERIC_LITERAL:
            if "." in token.value:
                return NumericConstant(token=token, value=float(token.value))
            return NumericConstant(token=token, value=int(token.value))
        # this represents an illegal invocation of _as_constant()
        raise AssertionError()

    def _as_id_constant(self, annotation: Token, constant: Identifier) -> IDConstant:
        if annotation.token_type == TokenType.EXCLAMATION_MARK:
            return IDConstant(
                token=constant, annotation=AnnotationType.BANG, value=constant.name
            )
        elif annotation.token_type == TokenType.QUESTION_MARK:
            return IDConstant(
                token=constant, annotation=AnnotationType.OPT, value=constant.name
            )
        # this represents an illegal invocation of _as_id_constant()
        raise AssertionError()

    def _parse_arguments(self) -> List[Union[Variable, Constant]]:
        self._check_token(TokenType.OPEN_PAREN, should_raise=True)

        arguments: List[Union[Variable, Constant]] = []
        while True:
            token = self._pop_token()
            if token is None:
                raise QueryLexerException(
                    SourceLocation(begin=self._eof, end=self._eof + 1),
                    "unterminated argument list",
                )
            if token.token_type == TokenType.CLOSE_PAREN:
                return arguments

            if token.token_type in [
                TokenType.EXCLAMATION_MARK,
                TokenType.QUESTION_MARK,
            ]:
                # We have an "annotated" identifier, e.g., !Q76 / ?Q76
                ident = self._pop_require_token()
                if ident.token_type != TokenType.IDENTIFIER:
                    raise AssertionError(
                        f"expected {TokenType.IDENTIFIER} but got {ident.token_type}"
                    )
                arguments.append(self._as_id_constant(annotation=token, constant=ident))
            elif token.token_type == TokenType.IDENTIFIER:
                arguments.append(Variable(token=token, name=token.name))
            elif token.token_type in [
                TokenType.STRING_LITERAL,
                TokenType.NUMERIC_LITERAL,
            ]:
                arguments.append(self._as_constant(token))
            else:
                raise QueryParserException(
                    token, f"unexpected {token.token_type} in argument list"
                )

            token = self._pop_token()
            if token is None:
                raise QueryLexerException(
                    SourceLocation(begin=self._eof, end=self._eof + 1),
                    "unterminated argument list",
                )
            if token.token_type == TokenType.CLOSE_PAREN:
                return arguments

            if token.token_type != TokenType.COMMA:
                raise QueryParserException(
                    token, f"unexpected {token.token_type} in argument list"
                )

    def _check_contains_colon(self) -> bool:
        colon_tokens = [
            *filter(lambda token: token.token_type == TokenType.COLON, self._tokens)
        ]
        if len(colon_tokens) > 1:
            raise QueryParserException(colon_tokens[1], "extranous ':' in query")
        return len(colon_tokens) == 1

    def _advance_cursor(self, amount=1) -> None:
        self._cursor += amount

    def _token_at(self, cursor) -> Optional[Token]:
        if cursor >= len(self._tokens):
            return None
        return self._tokens[cursor]

    def _require_token(self) -> Token:
        token = self._curr_token()
        if token is None:
            raise QueryLexerException(
                SourceLocation(begin=self._eof, end=self._eof + 1),
                "missing require token",
            )
        return token

    def _curr_token(self) -> Optional[Token]:
        return self._token_at(self._cursor)

    def _pop_require_token(self) -> Token:
        token = self._pop_token()
        if token is None:
            raise QueryLexerException(
                SourceLocation(begin=self._eof, end=self._eof + 1),
                "missing require token",
            )
        return token

    def _pop_token(self) -> Optional[Token]:
        token = self._curr_token()
        self._advance_cursor()
        return token

    def _check_token(self, token_type: TokenType, should_raise: bool = False) -> bool:
        token = self._pop_token()
        if token is None:
            if should_raise:
                raise QueryLexerException(
                    SourceLocation(begin=self._eof, end=self._eof + 1),
                    f"expected {token_type} but got EOF",
                )
            return False
        if token.token_type != token_type:
            if should_raise:
                raise QueryParserException(
                    token, f"expected {token_type} but got {token.token_type}"
                )
            return False
        return True

    def _peak_token(self):
        return self._token_at(self._cursor + 1)
