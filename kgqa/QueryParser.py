from typing import List, Dict, Tuple
import re
import traceback


class Var:
    def __init__(self, val):
        self.v = val

    def __eq__(self, other):
        if isinstance(other, Var):
            if other.v == self.v:
                return True
        return False

    def __hash__(self):
        return hash(self.v)

    def get_str(self):
        return "var_" + self.v.lower()

    def __repr__(self):
        return self.get_str()


def is_var(x):
    return isinstance(x, Var)


class ParsedQuery:
    def __init__(self, head=None, clauses=None, filters=[], expressq=""):
        # Head is a list of variables appearing in the head, e.g [Var.X, Var.Y]
        self.head = head

        # Clauses is a list of clauses. Each clause has the following structure:
        # [subj, predicate_sentence, obj]
        # where subj, obj are either a ""-string or a Var. abstract variable.
        self.clauses = clauses

        # Filters is a list of filters. Each filter has the followng structure.
        # [subj, [<>=], numerical expression]
        self.filters = filters

        self.expressq = expressq

    def is_valid(self):
        if type(self.clauses) != list:
            return False
        for c in self.clauses:
            if type(c) != str or len(c) != 3:
                return False
        return type(self.head) == list


def get_all_anchor_strings(pq: ParsedQuery):
    res = []
    for s, _, o in pq.clauses:
        if isinstance(s, str):
            res.append(s)
        if isinstance(o, str):
            res.append(o)
    return res


class QueryParser:
    def __init__(self):
        pass

    def get_parsed_clasue_string(self, single_clause_raw):
        """
        Given a clause of the form "starred_in_a_movie(X,Y)", where X or
        Y might be an anchor, returns a tuple:
        (subject, predicate_in_english, object)
        (X, "starred in a movie", Y)
        """
        # Strip last )
        single_clause_raw = single_clause_raw[:-1]

        # Get predicate, and s,o pair
        pred_raw, so_pair = single_clause_raw.split("(")

        # Replace underscores with spaces
        predicate_sentence = " ".join(pred_raw.split("_"))

        # Separate subject and object
        if "," in so_pair:
            # We have a binary predicate, e.g directed(M,D)
            subj, obj = so_pair.split(",")
            return [subj, predicate_sentence, obj]
        else:
            # We have a unary predicate, e.g movie(M)
            # Encode it as (subject, instance of, predicate)
            # In this example, (M, "instance of", "movie")
            return [so_pair, "instance of", f'"{predicate_sentence}"']

    def get_parsed_filter_from_arithmetic(self, arithmetic):
        x, op, const = arithmetic.split(" ")
        return [Var(x), op, int(const)]

    def parse(self, q: str):
        input_query = q
        # Preliminary parse
        try:
            heads_raw, body_raw = input_query.split(":")

            # Match all patterns that look like:
            # predicate(X,Y)
            # separated by ;
            clauses_raw = re.findall(r"[A-Za-z\_]+\([^;]+\)", body_raw)

            # Thanks chatGPT. First one (March 24, 2023).
            # I anticipate soon this codebase will be full of "thanks chatGPT!".
            arithmetic_raw = re.findall(
                r"(?:[A-Za-z]+)\s*[><=]+\s*(?:\d+(?:\.\d+)?[A-Za-z]*)", body_raw
            )

            # Split head variables
            assert "'" not in heads_raw
            heads = heads_raw.split(",")
            convert_head_tokens_to_Vars(heads)

            # Parse each raw clause string
            parsed_clauses, filters = [], []
            for clause_raw in clauses_raw:
                # Transform 'starred_in(X,Y)' --> [X, "starred in", Y]
                # or.       'movie(M)'        --> [M, "movie"]
                clause = self.get_parsed_clasue_string(clause_raw)
                clause = [x.strip() for x in clause]

                clause[0] = convert_to_Var_or_remove_double_quotes(clause[0])
                if len(clause) == 3:
                    # Binary
                    clause[2] = convert_to_Var_or_remove_double_quotes(clause[2])
                else:
                    pass
                parsed_clauses.append(clause)

            for arithmetic in arithmetic_raw:
                parsed_filter = self.get_parsed_filter_from_arithmetic(arithmetic)
                filters.append(parsed_filter)

            return ParsedQuery(
                head=heads, clauses=parsed_clauses, filters=filters, expressq=q
            )
        except Exception as e:
            print(f"Exception occured in parsing {e}")
            traceback.print_exc()
            return ParsedQuery([], [], [], "")


def is_double_quoted_string(s):
    """
    Returns true if s is a string and has no outside double quotes.
    is_double_quoted_string("\"some_string\"") = True
    is_double_quoted_string("\"\"") = True
    is_double_quoted_string("some_string") = False
    is_double_quoted_string("\"some_string") = False
    """
    return isinstance(s, str) and len(s) >= 2 and s[0] == '"' and s[-1] == '"'


def remove_double_quotes_if_present(s):
    if is_double_quoted_string(s):
        return s[1:-1]


def is_token(s):
    return isinstance(s, str) and len(s) == 1 and not is_double_quoted_string(s)


def convert_to_Var_or_remove_double_quotes(s):
    if is_token(s):
        return Var(s)
    elif is_double_quoted_string(s):
        return remove_double_quotes_if_present(s)
    else:
        print(
            f"Input {s} to a predicate is neither a token, nor a double quoted string."
        )
        assert False


def convert_head_tokens_to_Vars(lst: List[str]):
    n = len(lst)
    for i in range(n):
        if is_token(lst[i]):
            lst[i] = Var(lst[i])
