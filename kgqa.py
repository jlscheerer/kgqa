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
from kgqa.QueryParser import QueryParser, QueryParserException
from kgqa.PostProcessing import run_and_rank
from kgqa.MatchingUtils import compute_similar_entities, compute_similar_properties
from kgqa.FaissIndex import FaissIndexDirectory
from kgqa.Database import Database
from kgqa.SPARQLBackend import wqg2sparql
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

        backend = Preferences()["backend"]
        qs: QueryString
        if backend == "SQL":
            with yaspin(text="Emitting SQL Code..."):
                qs, stats = wqg2sql(wqg, stats)
        elif backend == "SPARQL":
            with yaspin(text="Emitting SPARQL Code..."):
                qs, stats = wqg2sparql(wqg, stats)
        else:
            raise AssertionError(
                f"trying to emit code for unknown backend: '{backend}'"
            )

        with yaspin(text="Executing Query on Wikidata..."):
            results, columns = run_and_rank(qs, wqg, stats)

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
    print(".property\t\tRetrieves similar properties")
    print(".search\t\t\tPerforms non-isomorphic search")
    print(".set\t\t\tSet user preferences")
    print(".exit\t\t\tExit this program")
    return True


def _handle_builtin_entity(args: List[str]) -> bool:
    db = Database()
    name = " ".join(args)
    with yaspin(text=f'Scanning for entities matching "{name}"...'):
        faiss = FaissIndexDirectory().labels
        qids, scores = compute_similar_entities(name)
        results = [
            (qid, score, faiss.label_for_id(qid), db.get_description_for_id(qid))
            for qid, score in zip(qids, scores)
        ]
    print(tabulate(results, ["QID", "Score", "Label", "Description"]))
    return True


def _handle_builtin_property(args: List[str]) -> bool:
    db = Database()
    name = " ".join(args)
    with yaspin(text=f'Scanning for predicates matching "{name}..."'):
        faiss = FaissIndexDirectory().properties
        pids, scores = compute_similar_properties(name)
        results = [
            (pid, score, faiss.label_for_id(pid), db.get_description_for_id(pid))
            for pid, score in zip(pids, scores)
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
