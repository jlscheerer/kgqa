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


# TODO(jlscheerer) Create one for Properties and one for Entities
class FaissIndex:
    def __init__(self, index, labels, ids):
        if index.ntotal != len(labels) or index.ntotal != len(ids):
            raise AssertionError("Attempting to construct invalid FaissIndex")
        self._index = index
        self._labels = labels
        self._ids = ids

        self._id2label = dict()
        for id, label in zip(self._ids, self._labels):
            self._id2label[id] = label

    def search(self, needle, count):
        scores, faiss_ids = self._index.search(
            np.array([Transformer().encode(needle)]), count
        )
        return {self._ids[id]: score for score, id in zip(scores[0], faiss_ids[0])}

    def label_for_id(self, id):
        return self._id2label[id]


class FaissIndexDirectory(metaclass=Singleton):
    def __init__(self):
        config = Config()
        self.labels = FaissIndex(
            load_pickle_from_file(
                config.file_in_directory("embeddings", FILENAME_FAISS_INDEX),
            ),
            read_partioned_strs("embeddings", "labels"),
            read_partioned_strs("embeddings", "qids"),
        )

        self.properties = FaissIndex(
            load_pickle_from_file(
                config.file_in_directory("embeddings", FILENAME_PROPERTY_FAISS)
            ),
            read_strs_from_file(
                config.file_in_directory("embeddings", FILENAME_PROPERTY_LABELS)
            ),
            read_strs_from_file(
                config.file_in_directory("embeddings", FILENAME_PROPERTY_IDS)
            ),
        )
