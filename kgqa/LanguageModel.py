from typing import Callable
import openai
import os
import json
import hashlib

from .Config import Config
from .Singleton import Singleton


class LMCache(metaclass=Singleton):
    def __init__(self):
        self._cache_depth = 1
        self._cache_path = Config().file_in_directory("language_model", "lmcache.json")
        if os.path.exists(self._cache_path):
            with open(self._cache_path, "r") as file:
                self._cache = json.load(file)
        else:
            self._cache = dict()

    def lookup_or_perform(self, prompt: str, fn: Callable[[str], str]) -> str:
        assert self._cache_depth == 1
        hash = self._hash(prompt)
        if hash in self._cache:
            return self._cache[hash]["results"][0]

        result = fn(prompt)
        self._cache[hash] = {"prompt": prompt, "results": [result]}
        self._flush()
        return result

    def _hash(self, prompt: str) -> str:
        return hashlib.md5(prompt.encode("utf-8")).hexdigest()

    def _flush(self):
        with open(self._cache_path, "w") as file:
            json.dump(self._cache, file)


class LanguageModel(metaclass=Singleton):
    def __init__(self):
        config = Config()["language_model"]
        openai.api_key = config["openai_api_key"]

    def complete(self, prompt: str):
        cache = LMCache()
        return cache.lookup_or_perform(prompt, self._complete_via_model)  # type: ignore

    def _complete_via_model(
        self, prompt: str, engine="gpt-3.5-turbo", response_length=1024, temperature=0.1
    ):
        try:
            response = openai.ChatCompletion.create(
                model=engine,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=response_length,
                temperature=temperature,
            )
            return response["choices"][0]["message"]["content"]
        except Exception as err:
            # TODO(jlscheerer) We would want to wrap this somehow.
            raise err
