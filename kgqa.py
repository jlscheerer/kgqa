import traceback
import pydoc
import readline  # noqa # pylint: disable=unused-import

from typing import List
from yaspin import yaspin
from tabulate import tabulate
from termcolor import colored
from kgqa.NonIsomorphicSearch import infer_n_hops_predicate

from kgqa.Preferences import Preferences

from kgqa.QueryBackend import QueryString
from kgqa.QueryGraph import query2aqg, aqg2wqg
from kgqa.QueryLexer import QueryLexerException, SourceLocation
from kgqa.QueryParser import (
    QueryParser,
    QueryParserException,
    QueryParserExceptionWithNote,
)

from kgqa.PostProcessing import run_and_rank
from kgqa.MatchingUtils import compute_similar_entities, compute_similar_properties
from kgqa.Database import Database
from kgqa.SPARQLBackend import wqg2sparql


def _display_query_results(results, columns):
    pydoc.pipepager(tabulate(results, columns, tablefmt="orgtbl"), cmd="less -R")


def _annotate_source(
    query: str, source_location: SourceLocation, type_: str, color: str, msg: str
):
    print(
        f"{colored(type_, color, attrs=['bold'])}: {colored(msg, 'white', attrs=['bold'])}"
    )
    print(f"  {query}")
    print(
        " " * (source_location.begin + 2)
        + colored("^" * (source_location.end - source_location.begin), "green")
    )


def _annotate_parser_error(query: str, source_location: SourceLocation, error: str):
    return _annotate_source(query, source_location, "error", "red", error)


def _annotate_parser_note(query: str, source_location: SourceLocation, note: str):
    return _annotate_source(query, source_location, "note", "yellow", note)


def _handle_user_query(query: str):
    try:
        with yaspin(text="Parsing Query..."):
            pq = QueryParser().parse(query)

        with yaspin(text="Generating Abstract Query Graph..."):
            aqg = query2aqg(pq)

        with yaspin(text="Synthesizing Executable Query Graph..."):
            wqg = aqg2wqg(aqg)

        backend = Preferences()["backend"]
        qs: QueryString
        if backend == "SPARQL":
            with yaspin(text="Emitting SPARQL Code..."):
                qs = wqg2sparql(wqg)
        else:
            raise AssertionError(
                f"trying to emit code for unknown backend: '{backend}'"
            )

        print(qs.value)
        print(qs.col2name)
        # with yaspin(text="Executing Query on Wikidata..."):
        #     results, columns = run_and_rank(qs, wqg)

        # _display_query_results(results, columns)
    except QueryLexerException as err:
        _annotate_parser_error(query, err.source_location, err.error)
        return
    except QueryParserException as err:
        _annotate_parser_error(query, err.token.source_location, err.error)
        return
    except Exception as err:
        print(f"Error: {err}")
        traceback.print_exc()


def _handle_user_help() -> bool:
    print(".entity\t\t\tRetrieves similar entities")
    print(".property\t\tRetrieves similar properties")
    print(".search\t\t\tPerforms non-isomorphic search")
    print(".set\t\t\tSet user preferences")
    print(".exit\t\t\tExit this program")
    return True


def _handle_builtin_entity(args: List[str]) -> bool:
    db = Database()
    name = " ".join(args)
    with yaspin(text=f'Scanning for entities matching "{name}"...'):
        qids, scores, labels = compute_similar_entities(name, return_labels=True)  # type: ignore
        results = [
            (qid, score, label, db.get_description_for_id(qid))
            for qid, score, label in zip(qids, scores, labels)
        ]
    print(tabulate(results, ["QID", "Score", "Label", "Description"]))
    return True


def _handle_builtin_property(args: List[str]) -> bool:
    db = Database()
    name = " ".join(args)
    with yaspin(text=f'Scanning for predicates matching "{name}"...'):
        pids, scores, labels = compute_similar_properties(name, return_labels=True)  # type: ignore
        results = [
            (pid, score, label, db.get_description_for_id(pid))
            for pid, score, label in zip(pids, scores, labels)
        ]
    print(tabulate(results, ["PID", "Score", "Label", "Description"]))
    return True


def _handle_builtin_search(args: List[str]) -> bool:
    # TODO(jlscheerer) Extend this scope of this builtin.
    try:
        assert len(args) == 2
        hops = int(args[0])
        predicate = args[1]

        infer_n_hops_predicate(predicate, n=hops)
    except:
        print("usage: .search <n-hops> <predicate>")
    return True


def _handle_builtin_parse(args: List[str]) -> bool:
    query = " ".join(args)
    try:
        qp = QueryParser()
        pq = qp.parse(
            query,
            lambda token, msg: _annotate_parser_note(query, token.source_location, msg),
        )
        print(pq.canonical())
    except QueryLexerException as err:
        _annotate_parser_error(query, err.source_location, err.error)
    except QueryParserException as err:
        _annotate_parser_error(query, err.token.source_location, err.error)
    except QueryParserExceptionWithNote as err:
        _annotate_parser_error(query, err.token.source_location, err.error)
        _annotate_parser_note(query, err.note_token.source_location, err.note)
    except Exception as err:
        print(f"Error: {err}")
        traceback.print_exc()

    return True


def _handle_builtin_set(args: List[str]) -> bool:
    if len(args) != 2:
        print(
            f"{colored('error', 'red', attrs=['bold'])}: {colored('illegal format for .set <key> <value>', 'white', attrs=['bold'])}"
        )
    else:
        try:
            Preferences().set(args[0], args[1])
        except AssertionError as err:
            print(
                f"{colored('error', 'red', attrs=['bold'])}: {colored(err.args[0], 'white', attrs=['bold'])}"
            )
    return True


def _handle_user_builtin(command: str) -> bool:
    builtin, *args = command.split()
    if builtin == ".exit":
        return False
    elif builtin == ".help":
        return _handle_user_help()
    elif builtin == ".entity":
        return _handle_builtin_entity(args)
    elif builtin == ".property":
        return _handle_builtin_property(args)
    elif builtin == ".search":
        return _handle_builtin_search(args)
    elif builtin == ".parse":
        return _handle_builtin_parse(args)
    elif builtin == ".set":
        return _handle_builtin_set(args)
    print(
        f'Error: unknown command or invalid arguments: "{builtin}". Enter ".help" for help'
    )
    return True


def _handle_user_command(command: str) -> bool:
    if command.strip().startswith("."):
        return _handle_user_builtin(command)
    _handle_user_query(command)
    return True


def main():
    intc = 0
    while True:
        try:
            command = input("> ")
            if not _handle_user_command(command.strip()):
                break
            intc = 0
        except KeyboardInterrupt:
            print()
            intc += 1
            if intc > 1:
                print(colored("aborting session due to keyboard interrupt", "red"))
                break


if __name__ == "__main__":
    main()
