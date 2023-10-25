import os
import json
import re
from typing import Any, Callable, Dict

from kgqa.Config import Config
from kgqa.LanguageModel import LanguageModel
from kgqa.Singleton import Singleton


class PromptTemplate:
    def __init__(self, template: str, response_parser: Callable[[str], Any]):
        self._template = template
        self._vars = set(re.findall("\\${(.*?)}", self._template))
        self._response_parser = response_parser

    def requires_var(self, var: str) -> bool:
        return var in self._vars


def parse_semicolon_separated_lines(response: str):
    result = []
    lines = response.splitlines()
    for line in lines:
        if not re.match("^[^;]*;[^;]*$", line):
            raise AssertionError(f"received unexpected prompt response: {response}")
        split = line.split(";")
        result.append(tuple([x.strip() for x in split]))
    return result


class PromptTemplateDirectory(metaclass=Singleton):
    def __init__(self):
        config = Config()
        with open(
            config.file_in_directory("prompt_builder", "prompts.json"), "r"
        ) as file:
            prompts = json.load(file)

        self._parsers = {"semicolon_separated_lines": parse_semicolon_separated_lines}
        self._prompts = dict()
        for name, prompt in prompts.items():
            with open(
                os.path.join(config.directory("prompt_builder"), prompt["template"]),
                "r",
            ) as file:
                template = file.read()

            self._prompts[name] = PromptTemplate(
                template, self._parsers[prompt["response"]]
            )

    def template_with_name(self, name: str):
        return self._prompts[name]


class PromptBuilder:
    def __init__(self, template: str):
        self._template = PromptTemplateDirectory().template_with_name(template)
        self._values: Dict[str, str] = dict()

    def set(self, var: str, value: str):
        if var in self._values or not self._template.requires_var(var):
            raise AssertionError("attempting to assign to unexpected variable.")
        self._values[var] = value
        return self

    def serialize(self) -> str:
        prompt = self._template._template
        for var in self._template._vars:
            if var not in self._values:
                raise AssertionError(f"missing value for variable '{var}'")
            prompt = prompt.replace(f"${{{var}}}", self._values[var])
        return prompt

    def execute(self):
        lm = LanguageModel()
        return self._template._response_parser(lm.complete(self.serialize()))
