from typing import List, Tuple
from kgqa.FaissIndex import FaissIndexDirectory

# TODO(jlscheerer) Reimplment the LanguageModel follow up logic and alias handling.
from .Config import Config


# TODO(jlscheerer) Reimplment alias handling logic here.
def compute_similar_predicates(predicate: str) -> Tuple[List[str], List[float]]:
    """
    For a given predicate specified in english, returns a list of predicates,
    their ids, and assoc. probs in the form:
        [prob:int, pid:str, label:str]
    where each probability is the cosine similarity (inner product) with the given predicate.

    Returns top-num_preds such entities.
    """
    config = Config()
    pid_to_score = FaissIndexDirectory().properties.search(
        predicate, config["NumTopPID"]
    )
    pids_scores = sorted(
        [(pid, score) for pid, score in pid_to_score.items()],
        key=lambda x: x[1],
        reverse=True,
    )
    pids, scores = zip(*pids_scores)
    return pids, scores  # type: ignore


# TODO(jlscheerer) Reimplment alias handling logic here.
def compute_similar_entity_ids(entity: str) -> Tuple[List[str], List[float]]:
    """
    For a given entity specified in english, returns a list of lists of the follwing form:
        [prob:int, qid:str, label:str, id:int]
    where each probability is the cosine similarity (inner product) with the given entity.

    Returns top-num_qids such entities.
    """
    config = Config()
    num_qids = config["NumTopQID"]
    pid_to_score = FaissIndexDirectory().labels.search(entity, num_qids)
    pids_scores = sorted(
        [(pid, score) for pid, score in pid_to_score.items()],
        key=lambda x: x[1],
        reverse=True,
    )
    pids, scores = zip(*pids_scores)
    return pids, scores  # type: ignore
