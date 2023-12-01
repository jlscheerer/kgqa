import faiss
import numpy as np
from math import exp
import pandas as pd

from .Constants import (
    FILENAME_FAISS_INDEX,
    FILENAME_PROPERTY_FAISS,
)
from .Singleton import Singleton
from .Config import Config
from .Transformers import Transformer
from .Database import Database

# NOTE Ranking weight contributions.
LABEL_WEIGHT = 0.55
DESCRIPTION_WEIGHT = 0.15
POPULARITY_WEIGHT = 0.3

POPULARITY_SCALE = 100


def faiss_id_to_int(id):
    assert id[0] in ["P", "Q"]
    val = int(id[1:])
    # NOTE use lsb to indicate P/Q
    return 2 * val + (1 if id[0] == "P" else 0)


def faiss_int_to_id(val):
    p_q = "P" if (val % 2 == 1) else "Q"
    return f"{p_q}{val // 2}"


def sigmoid(x):
    return 1 / (1 + exp(-x))


class FaissIndex:
    def __init__(self, index):
        config = Config()
        self._index = faiss.read_index(config.file_in_directory("embeddings", index))

    def search(self, needle, count):
        # TODO(jlscheerer) Support batching queries.
        faiss_scores, faiss_ids = self._index.search(
            np.array([Transformer().encode(needle)]), count
        )
        faiss_scores, faiss_ids = faiss_scores[0], faiss_ids[0]
        ids = [faiss_int_to_id(id) for id in faiss_ids]
        meta = self._retrieve_meta(ids)

        scores, pscores, dscores = [], [], []
        query = Transformer().encode(needle)
        for index, id in enumerate(ids):
            faiss_score = faiss_scores[index]
            # TODO(jlscheerer) Perform this in batches also.
            if meta[id]["description"] is not None:
                description_score = np.inner(
                    query, Transformer().encode(meta[id]["description"])
                )
            else:
                description_score = 0.125
            popularity_score = self._popularity_score(meta[id]["popularity"])
            dscores.append(description_score)
            pscores.append(popularity_score)
            scores.append(
                LABEL_WEIGHT * faiss_score
                + DESCRIPTION_WEIGHT * description_score
                + POPULARITY_WEIGHT * popularity_score
            )

        df = pd.DataFrame(
            {
                "id": ids,
                "label": [meta[id]["label"] for id in ids],
                "score": scores,
                "faiss": faiss_scores,
                "pscore": pscores,
                "dscore": dscores,
                "description": [meta[id]["description"] for id in ids],
            }
        )
        df.sort_values(by=["score"], ascending=False, inplace=True)

        print(df)

        results = dict()
        for rank, (index, row) in enumerate(df.iterrows()):
            if rank >= 5:
                break
            results[row["id"]] = row["score"]

        return list(df["id"][:5]), list(df["label"][:5]), list(df["score"][:5])

    def _popularity_score(self, popularity):
        return sigmoid(popularity / POPULARITY_SCALE)

    def _retrieve_meta(self, ids):
        db = Database()
        entity_ids = ", ".join([f"'{id}'" for id in ids])
        meta_data_rows = db.fetchall(
            f"""
        SELECT l.id, l.value, d.value, p.count
        FROM labels_en l LEFT JOIN descriptions_en d   ON (l.id = d.id) 
                         LEFT JOIN entity_popularity p ON(l.id = p.entity_id)
        WHERE entity_id IN ({entity_ids})
        """
        )
        return {
            id: {"label": label, "description": description, "popularity": popularity}
            for id, label, description, popularity in meta_data_rows
        }

    def label_for_id(self, id):
        raise AssertionError


class FaissIndexDirectory(metaclass=Singleton):
    def __init__(self):
        self.labels = FaissIndex(FILENAME_FAISS_INDEX)
        self.properties = FaissIndex(FILENAME_PROPERTY_FAISS)
