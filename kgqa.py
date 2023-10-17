import traceback
import pydoc
import readline  # noqa # pylint: disable=unused-import

from typing import List
from yaspin import yaspin
from tabulate import tabulate
from termcolor import colored

from kgqa.Preferences import Preferences
from kgqa.QueryGraph import query2aqg, aqg2wqg
from kgqa.QueryLexer import QueryLexerException, SourceLocation
from kgqa.QueryParser import QueryParser, QueryParserException
from kgqa.PostProcessing import run_and_rank
from kgqa.MatchingUtils import compute_similar_entity_ids, compute_similar_predicates
from kgqa.FaissIndex import FaissIndexDirectory
from kgqa.Database import Database
from kgqa.SQLBackend import wqg2sql


def _display_query_results(results, columns):
    pydoc.pipepager(tabulate(results, columns, tablefmt="orgtbl"), cmd="less -R")


def _annotate_parser_error(query: str, source_location: SourceLocation, error: str):
    print(
        f"{colored('error', 'red', attrs=['bold'])}: {colored(error, 'white', attrs=['bold'])}"
    )
    print(f"  {query}")
    print(
        " " * (source_location.begin + 2)
        + colored("^" * (source_location.end - source_location.begin), "green")
    )


def _handle_user_query(query: str):
    try:
        with yaspin(text="Parsing Query..."):
            pq = QueryParser().parse(query)

        with yaspin(text="Generating Abstract Query Graph..."):
            aqg, stats = query2aqg(pq)

        with yaspin(text="Synthesizing Executable Query Graph..."):
            wqg, stats = aqg2wqg(aqg, stats)

        with yaspin(text="Emitting SQL Code..."):
            sql, stats = wqg2sql(wqg, stats)

        with yaspin(text="Executing Query on Wikidata..."):
            results, columns = run_and_rank(sql, wqg, stats)

        _display_query_results(results, columns)
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
    print(".predicate\t\tRetrieves similar predicates")
    print(".search\t\t\tPerforms non-isomorphic search")
    print(".set\t\t\tSet user preferences")
    print(".exit\t\t\tExit this program")
    return True


def _handle_builtin_entity(args: List[str]) -> bool:
    db = Database()
    name = " ".join(args)
    with yaspin(text=f'Scanning for entities matching "{name}"...'):
        faiss = FaissIndexDirectory().labels
        qids, scores = compute_similar_entity_ids(name)
        results = [
            (qid, score, faiss.label_for_id(qid), db.get_description_for_id(qid))
            for qid, score in zip(qids, scores)
        ]
    print(tabulate(results, ["QID", "Score", "Label", "Description"]))
    return True


def _handle_builtin_predicate(args: List[str]) -> bool:
    db = Database()
    name = " ".join(args)
    with yaspin(text=f'Scanning for predicates matching "{name}..."'):
        faiss = FaissIndexDirectory().properties
        pids, scores = compute_similar_predicates(name)
        results = [
            (pid, score, faiss.label_for_id(pid), db.get_description_for_id(pid))
            for pid, score in zip(pids, scores)
        ]
    print(tabulate(results, ["PID", "Score", "Label", "Description"]))
    return True


def _handle_builtin_search(args: List[str]) -> bool:
    # TODO(jlscheerer) Perform non-isomorphic search
    print(".search NYI:", args)
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
    elif builtin == ".predicate":
        return _handle_builtin_predicate(args)
    elif builtin == ".search":
        return _handle_builtin_search(args)
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
    while True:
        command = input("> ")
        if not _handle_user_command(command.strip()):
            break


if __name__ == "__main__":
    main()
