from typing import List, Tuple

from .Config import Config
from .RequestServer import RequestServer, request_handler
from .FaissIndex import FaissIndexDirectory


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


class FaissServer(RequestServer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @request_handler
    def compute_similar_properties(self, data):
        predicate = data["property"]
        num_pids = data["num_pids"]
        pid_to_score = FaissIndexDirectory().properties.search(predicate, num_pids)
        pids_scores = sorted(
            [(pid, score) for pid, score in pid_to_score.items()],
            key=lambda x: x[1],
            reverse=True,
        )
        pids, scores = zip(*pids_scores)
        labels = [FaissIndexDirectory().labels.label_for_id(id) for id in pids]
        self.write(
            {
                "pids": pids,
                "scores": [float(x) for x in scores],
                "labels": labels,
            }
        )

    @request_handler
    def compute_similar_entities(self, data):
        entity = data["entity"]
        num_qids = data["num_qids"]
        pid_to_score = FaissIndexDirectory().labels.search(entity, num_qids)
        pids_scores = sorted(
            [(pid, score) for pid, score in pid_to_score.items()],
            key=lambda x: x[1],
            reverse=True,
        )
        qids, scores = zip(*pids_scores)
        labels = [FaissIndexDirectory().labels.label_for_id(id) for id in qids]
        self.write(
            {"qids": qids, "scores": [float(x) for x in scores], "labels": labels}
        )


if __name__ == "__main__":
    _ = FaissIndexDirectory()
    print("FaissServer successfully loaded Index Directory.")
    FaissServer.start_server("127.0.01", 43096)
