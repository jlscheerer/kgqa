from sentence_transformers import SentenceTransformer, util

from .Singleton import Singleton
from .Config import Config


class Transformer(metaclass=Singleton):
    def __init__(self):
        config = Config()["embeddings"]["transformer"]
        self._model = SentenceTransformer(config["model"], device=config["device"])

    def encode(self, words):
        return self._model.encode(words)
