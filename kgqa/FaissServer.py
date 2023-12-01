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
    debug_mode: bool = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @request_handler
    def compute_similar_properties(self, data):
        if FaissServer.debug_mode:
            self.write(
                {"pids": ["P123"], "scores": [1.0], "labels": ["Debug Property P123"]}
            )
            return
        predicate = data["property"]
        num_pids = data["num_pids"]
        ids, labels, scores = FaissIndexDirectory().properties.search(
            predicate, num_pids
        )
        self.write({"pids": ids, "scores": scores, "labels": labels})

    @request_handler
    def compute_similar_entities(self, data):
        if FaissServer.debug_mode:
            self.write(
                {"qids": ["Q123"], "scores": [1.0], "labels": ["Debug Entity Q123"]}
            )
            return

        entity = data["entity"]
        num_qids = data["num_qids"]
        ids, labels, scores = FaissIndexDirectory().labels.search(entity, num_qids)
        self.write({"qids": ids, "scores": scores, "labels": labels})


if __name__ == "__main__":
    _ = FaissIndexDirectory()
    print("FaissServer successfully loaded Index Directory.")
    FaissServer.start_server("127.0.01", 43096)
