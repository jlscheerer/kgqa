import yaml
import os

from kgqa.Singleton import Singleton

DEFAULT_PREFERENCES_PATH = "./preferences.yaml"
DEFAULT_PREFERENCES = {
    "backend": {"value": "SPARQL", "allowed": ["SPARQL"]},
    "print_sparql": {"value": "false", "allowed": ["true", "false"]},
    "print_sql": {"value": "false", "allowed": ["true", "false"]},
}


class Preferences(metaclass=Singleton):
    def __init__(self):
        if not os.path.exists(DEFAULT_PREFERENCES_PATH):
            self._initialize_config()
        with open(DEFAULT_PREFERENCES_PATH, "r") as file:
            preferences = yaml.safe_load(file)
        self._preferences = preferences

    def __getitem__(self, item):
        try:
            return self._preferences[item]["value"]
        except Exception:
            raise AssertionError(f"preferences is missing required parameter: {item}")

    def set(self, key, value):
        if key not in self._preferences:
            raise AssertionError(f"trying to update illegal key: '{key}'")
        if value not in self._preferences[key]["allowed"]:
            raise AssertionError(f"illegal value '{value}' for key '{key}'")
        self._preferences[key]["value"] = value
        self._flush()

    def _initialize_config(self):
        self._preferences = DEFAULT_PREFERENCES
        self._flush()

    def _flush(self):
        with open(DEFAULT_PREFERENCES_PATH, "w") as file:
            yaml.dump(self._preferences, file)
