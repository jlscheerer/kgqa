from typing import List, Tuple

from .Config import Config
from .RequestServer import RequestServer, request_handler
from .FaissIndex import FaissIndexDirectory


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
def compute_similar_entity_ids(
    entity: str, num_qids=None
) -> Tuple[List[str], List[float]]:
    """
    For a given entity specified in english, returns a list of lists of the follwing form:
        [prob:int, qid:str, label:str, id:int]
    where each probability is the cosine similarity (inner product) with the given entity.

    Returns top-num_qids such entities.
    """
    config = Config()
    if num_qids is None:
        num_qids = config["NumTopQID"]
    pid_to_score = FaissIndexDirectory().labels.search(entity, num_qids)
    pids_scores = sorted(
        [(pid, score) for pid, score in pid_to_score.items()],
        key=lambda x: x[1],
        reverse=True,
    )
    pids, scores = zip(*pids_scores)
    return pids, scores  # type: ignore


class FaissServer(RequestServer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @request_handler
    def compute_similar_properties(self, data):
        predicate = data["property"]
        # TODO(jlscheerer) Rename the function in MatchingUtils.
        pids, scores = compute_similar_predicates(predicate)
        self.write({"pids": pids, "scores": [float(x) for x in scores]})

    @request_handler
    def compute_similar_entities(self, data):
        entity = data["entity"]
        # TODO(jlscheerer) Rename the function in MatchingUtils and clean it up.
        qids, scores = compute_similar_entity_ids(entity)
        self.write({"qids": qids, "scores": [float(x) for x in scores]})


if __name__ == "__main__":
    _ = FaissIndexDirectory()
    print("FaissServer successfully loaded Index Directory.")
    FaissServer.start_server("127.0.01", 43096)
