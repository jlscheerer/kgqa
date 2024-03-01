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

def prepare_property(property):
    return " ".join(property.split("_"))

_entity_cache: dict[Tuple[str, int], Tuple[any, any, any]] = dict()
_property_cache: dict[Tuple[str, int], Tuple[any, any, any]] = dict()

class FaissServer(RequestServer):
    debug_mode: bool = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @request_handler
    def compute_similar_properties(self, data):
        global _property_cache

        if FaissServer.debug_mode:
            self.write(
                {"pids": ["P123"], "scores": [1.0], "labels": ["Debug Property P123"]}
            )
            return
        
        property = prepare_property(data["property"])
        num_pids = data["num_pids"]
        key = (property, num_pids)
        if key in _property_cache:
            ids, labels, scores = _property_cache[key]
        else:
            ids, labels, scores = FaissIndexDirectory().properties.search(
                property, num_pids, count_summary=3
            )
            _property_cache[key] = (ids, labels, scores)
        self.write({"pids": ids, "scores": scores, "labels": labels})

    @request_handler
    def compute_similar_entities(self, data):
        global _entity_cache

        if FaissServer.debug_mode:
            self.write(
                {"qids": ["Q123"], "scores": [1.0], "labels": ["Debug Entity Q123"]}
            )
            return

        entity = data["entity"]
        num_qids = data["num_qids"]
        key = (entity, num_qids)
        if key in _entity_cache:
            ids, labels, scores = _entity_cache[key]
        else:
            ids, labels, scores = FaissIndexDirectory().labels.search(
                entity, num_qids, count_summary=5
            )
            _entity_cache[key] = (ids, labels, scores)
        self.write({"qids": ids, "scores": scores, "labels": labels})


if __name__ == "__main__":
    _ = FaissIndexDirectory()
    print("FaissServer successfully loaded Index Directory.")
    FaissServer.start_server("127.0.01", 43096)
