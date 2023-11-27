from typing import List, Tuple

from .RequestServer import RequestClient
from .Config import Config


def compute_similar_entities(
    entity: str, num_qids=None
) -> Tuple[List[str], List[float]]:
    """
    For a given entity specified in english, returns a list of lists of the follwing form,
    where each probability is the cosine similarity (inner product) with the given entity.

    Returns num_qids such entities.
    """
    config = Config()
    if num_qids is None:
        num_qids = config["NumTopQID"]
    server = config["embeddings"]["server"]
    with RequestClient(server["host"], server["port"]) as client:
        response = client.compute_similar_entities(entity=entity, num_qids=num_qids)
        return response["qids"], response["scores"]


def compute_similar_properties(
    property: str, num_pids=None
) -> Tuple[List[str], List[float]]:
    """
    For a given property specified in english, returns a list of properties,
    where each probability is the cosine similarity (inner product) with the given property.

    Returns num_pids such properties.
    """
    config = Config()
    if num_pids is None:
        num_pids = config["NumTopPID"]
    server = config["embeddings"]["server"]
    with RequestClient(server["host"], server["port"]) as client:
        response = client.compute_similar_properties(
            property=property, num_pids=num_pids
        )
        return response["pids"], response["scores"]
