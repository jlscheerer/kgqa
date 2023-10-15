from enum import Enum, auto
from typing import List, Union
from .QueryLexer import (
    Identifier,
    LiteralToken,
    NumericLiteral,
    QueryLexer,
    QueryLexerException,
    SourceLocation,
    StringLiteral,
    Token,
    TokenType,
)
from dataclasses import dataclass


@dataclass
class Variable:
    token: Identifier
    name: str

    def __repr__(self):
        return f"Variable({self.name})"


@dataclass
class Constant:
    token: LiteralToken


@dataclass
class StringConstant(Constant):
    token: StringLiteral
    value: str

    def __repr__(self):
        return f"Constant('{self.value}')"


@dataclass
class NumericConstant(Constant):
    token: NumericLiteral
    value: Union[int, float]

    def __repr__(self):
        return f"Constant({self.value})"


class AggregationType(Enum):
    COUNT = auto()
    SUM = auto()
    AVG = auto()
    MAX = auto()
    MIN = auto()


@dataclass
class Aggregation:
    var: Variable
    type_: AggregationType

    def __repr__(self):
        return f"{self.type_.name}({self.var})"


HeadItem = Union[Variable, Aggregation]
QueryHead = List[HeadItem]


@dataclass
class QueryAtom:
    pass


@dataclass
class Predicate(QueryAtom):
    predicate: Variable
    arguments: List[Union[Variable, Constant]]

    def __repr__(self):
        return (
            f"{self.predicate.name}({', '.join([str(arg) for arg in self.arguments])})"
        )


class FilterOp(Enum):
    EQ = "="
    NEQ = "!="
    LE = "<"
    LEQ = "<="
    GE = ">"
    GEQ = ">="


@dataclass
class Filter(QueryAtom):
    lhs: Variable
    op: FilterOp
    rhs: Constant

    def __repr__(self):
        return f"{self.lhs} {self.op.value} {self.rhs}"


@dataclass
class ParsedQuery:
    head: QueryHead
    body: List[QueryAtom]


class QueryParserException(Exception):
    def __init__(self, token: Token, error: str):
        self.token = token
        self.error = error


class QueryParser:
    _tokens: List[Token]
    _cursor: int
    _eof: int

    def parse(self, query: str):
        self._tokens = list(QueryLexer(query))
        self._cursor = 0
        self._eof = len(query)
        head: QueryHead = []
        if self._check_contains_colon():
            head = self._parse_head()
        body = self._parse_body()
        return ParsedQuery(head=head, body=body)

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
            item = Aggregation(var=var, type_=AggregationType[token.name])
        else:
            raise QueryParserException(
                token, f"unexpected {token.token_type} in query head"
            )

        token = self._pop_token()
        # We should always at least encounter the `COLON` token.
        assert token is not None
        if token.token_type == TokenType.COMMA:
            return [item] + self._parse_head()
        elif token.token_type != TokenType.COLON:
            raise QueryParserException(
                token, f"unexpected {token.token_type} in query head"
            )
        return [item]

    def _parse_body(self) -> List[QueryAtom]:
        atoms: List[QueryAtom] = []
        while True:
            atom = self._parse_atom()
            if atom is None:
                raise QueryLexerException(
                    SourceLocation(begin=self._eof, end=self._eof + 1),
                    "missing query body",
                )
            atoms.append(atom)
            if self._curr_token() is None:
                break
            token = self._pop_require_token()
            if token.token_type != TokenType.SEMICOLON:
                raise QueryParserException(
                    token, f"unexpected {token.token_type} in query body"
                )
        return atoms

    def _parse_atom(self):
        ident = self._pop_token()
        if ident is None:
            return None

        if ident.token_type != TokenType.IDENTIFIER:
            raise QueryParserException(
                ident, f"unexpected {ident.token_type} in query body"
            )
        token = self._require_token()
        if token.token_type == TokenType.OPEN_PAREN:
            arguments = self._parse_arguments()
            return Predicate(
                predicate=Variable(token=ident, name=ident.name), arguments=arguments
            )
        elif token.token_type == TokenType.COMPARATOR:
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
            return Filter(
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
        # this represents an illegal invokation of _as_constant()
        raise AssertionError()

    def _parse_arguments(self):
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

            if token.token_type == TokenType.IDENTIFIER:
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

    def _check_contains_colon(self):
        colon_tokens = [
            *filter(lambda token: token.token_type == TokenType.COLON, self._tokens)
        ]
        if len(colon_tokens) > 1:
            raise QueryParserException(colon_tokens[1], "extranous ':' in query")
        return len(colon_tokens) == 1

    def _advance_cursor(self, amount=1):
        self._cursor += amount

    def _token_at(self, cursor):
        if cursor >= len(self._tokens):
            return None
        return self._tokens[cursor]

    def _require_token(self):
        token = self._curr_token()
        if token is None:
            raise QueryLexerException(
                SourceLocation(begin=self._eof, end=self._eof + 1),
                "missing require token",
            )
        return token

    def _curr_token(self):
        return self._token_at(self._cursor)

    def _pop_require_token(self):
        token = self._pop_token()
        if token is None:
            raise QueryLexerException(
                SourceLocation(begin=self._eof, end=self._eof + 1),
                "missing require token",
            )
        return token

    def _pop_token(self):
        token = self._curr_token()
        self._advance_cursor()
        return token

    def _check_token(self, token_type: TokenType, should_raise: bool = False):
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
