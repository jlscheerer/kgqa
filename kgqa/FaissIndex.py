import faiss
import numpy as np

from .Constants import (
    FILENAME_FAISS_INDEX,
    FILENAME_PROPERTY_IDS,
    FILENAME_PROPERTY_LABELS,
    FILENAME_PROPERTY_FAISS,
)
from .FileUtils import load_pickle_from_file, read_partioned_strs, read_strs_from_file
from .Singleton import Singleton
from .Config import Config
from .Transformers import Transformer


def faiss_id_to_int(id):
    assert id[0] in ["P", "Q"]
    val = int(id[1:])
    # NOTE use lsb to indicate P/Q
    return 2 * val + (1 if id[0] == "P" else 0)


def faiss_int_to_id(val):
    p_q = "P" if (val % 2 == 1) else "Q"
    return f"{p_q}{val // 2}"


# TODO(jlscheerer) Create one for Properties and one for Entities
class FaissIndex:
    def __init__(self, index):
        config = Config()
        self._index = faiss.read_index(config.file_in_directory("embeddings", index))

    def search(self, needle, count):
        scores, faiss_ids = self._index.search(
            np.array([Transformer().encode(needle)]), count
        )
        return {self._ids[id]: score for score, id in zip(scores[0], faiss_ids[0])}

    def label_for_id(self, id):
        raise AssertionError


class FaissIndexDirectory(metaclass=Singleton):
    def __init__(self):
        self.labels = FaissIndex(FILENAME_FAISS_INDEX)
        self.properties = FaissIndex(FILENAME_PROPERTY_FAISS)
