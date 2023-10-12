import openai

from .Config import Config
from .Singleton import Singleton


class LanguageModel(metaclass=Singleton):
    def __init__(self):
        config = Config()["language_model"]
        openai.api_key = config["openai_api_key"]

    def complete(self, input: str):
        # TODO(jlscheerer) Implement a persistent cache over input.
        return self._complete_via_model(input)

    def _complete_via_model(
        self, prompt, engine="gpt-3.5-turbo", response_length=1024, temperature=0.1
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
            raise err
