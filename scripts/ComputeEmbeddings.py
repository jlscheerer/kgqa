import os
import re
from collections import defaultdict

import faiss
import json
import numpy as np
import pandas as pd
from yaspin import yaspin
import pickle

from kgqa.Config import Config
from kgqa.Database import Database
from kgqa.FileUtils import partioned_files_in_directory
from kgqa.Transformers import Transformer
from kgqa.Constants import *


def _dump_strs_to_file(filename, data):
    with open(filename, "w") as file:
        file.writelines([f"{x}\n" for x in data])


def _read_strs_from_file(filename):
    with open(filename, "r") as file:
        return [x.strip() for x in file.readlines()]


def _dump_dict_to_file(filename, data):
    with open(filename, "w") as file:
        json.dump(data, file)


def _dump_pickle_to_file(filename, data):
    with open(filename, "wb") as file:
        pickle.dump(data, file)


# TODO(jlscheerer) Discuss with @Anton, this overlaps with the properties? Why?
def extract_db_labels():
    config = Config()
    db = Database()

    sql = f"""
    SELECT id, value
    FROM labels
    WHERE language = 'en'
    ORDER BY substring(id from '[0-9]+$')::int ASC
    LIMIT {config['embeddings']['count']};
    """

    BATCH_SIZE = config["embeddings"]["batch_size"]
    with yaspin(text="Extracting Labels...") as sp:
        for index, rows in enumerate(db.fetchmany(sql, BATCH_SIZE)):
            low, high = index * BATCH_SIZE, index * BATCH_SIZE + len(rows) - 1
            ids, labels = zip(*rows)

            ids_file = config.file_in_directory("embeddings", f"qids_{low}_{high}.txt")
            _dump_strs_to_file(ids_file, ids)

            labels_file = config.file_in_directory(
                "embeddings", f"labels_{low}_{high}.txt"
            )
            _dump_strs_to_file(labels_file, labels)

            sp.green.ok("✔ ")


def compute_label_embeddings():
    config = Config()

    with yaspin(text="Computing Embeddings...") as sp:
        for file in os.listdir(config.directory("embeddings")):
            if not re.match("^labels_[0-9]+_[0-9]+\\.txt$", file):
                continue
            labels = _read_strs_from_file(
                os.path.join(config.directory("embeddings"), file)
            )
            low, high = [*map(int, file[len("labels_") : -len(".txt")].split("_"))]

            embeddings_file = config.file_in_directory(
                "embeddings", f"emb_{low}_{high}.npy"
            )
            embeddings = Transformer().encode(labels)
            np.save(embeddings_file, embeddings)

            sp.green.ok("✔ ")


def extract_db_aliases_en():
    config = Config()
    db = Database()

    sql = f"""
    SELECT id, value
    FROM aliases
    WHERE language = 'en'
      AND substring(id from '[0-9]+$')::int < {config['embeddings']['alias_count']} 
    ORDER BY substring(id from '[0-9]+$')::int ASC
    """

    with yaspin(text="Extracting Aliases...") as sp:
        aliases = defaultdict(lambda: list())
        for key, value in db.fetchall(sql):
            aliases[key].append(value)

        aliases_file = config.file_in_directory("embeddings", "aliases_en.json")
        _dump_dict_to_file(aliases_file, dict(aliases))

        sp.green.ok("✔ ")


def _strip_aliases_dict(aliases_dict):
    def valid_alias(alias):
        return bool(re.match("^[a-zA-Z\\s]+$", alias))

    return {
        key: [*filter(valid_alias, value)]
        for key, value in aliases_dict.items()
        if len([*filter(valid_alias, value)]) > 0
    }


def compute_aliases_embeddings():
    config = Config()

    with yaspin(text="Computing Aliases Embeddings...") as sp:
        aliases_file = config.file_in_directory("embeddings", "aliases_en.json")
        with open(aliases_file, "r") as file:
            aliases_dict = json.load(file)

        aliases_dict = _strip_aliases_dict(aliases_dict)
        df = pd.DataFrame(
            [(key, value) for key, values in aliases_dict.items() for value in values]
        )

        index_to_qid_file = config.file_in_directory(
            "embeddings", "aliases_index_to_qid.json"
        )
        index_to_qid = {index: qid for index, qid in zip(df.index, df[0])}
        _dump_dict_to_file(index_to_qid_file, index_to_qid)

        # TODO(jlscheerer) This seems unnecessary, check with @Anton.
        df = df[df[0] > "Q"]
        aliases_only = np.array(df[1])

        embeddings = Transformer().encode(aliases_only)
        embeddings_file = config.file_in_directory(
            "embeddings", "aliases_embeddings.npy"
        )
        np.save(embeddings_file, embeddings)

        sp.green.ok("✔ ")


def _get_label_embeddings_files():
    return partioned_files_in_directory("embeddings", "emb", "npy")


def _load_label_embeddings():
    config = Config()
    return np.concatenate(
        [
            np.load(config.file_in_directory("embeddings", label_embeddings_file))
            for label_embeddings_file in _get_label_embeddings_files()
        ]
    )


# TODO(jlscheerer) Think about integrating the aliases directly into the index.
def compute_faiss_index():
    config = Config()

    with yaspin(text="Computing Faiss Index...") as sp:
        # aliases_embeddings = np.load(
        #     config.file_in_directory("embeddings", "aliases_embeddings.npy")
        # )
        embeddings = _load_label_embeddings()

        # embeddings = np.concatenate([label_embeddings, aliases_embeddings], axis=0)
        dimensions = embeddings.shape[1]
        factory_settings = "Flat"

        idx = faiss.index_factory(
            dimensions, factory_settings, faiss.METRIC_INNER_PRODUCT
        )
        idx.add(embeddings)
        _dump_pickle_to_file(
            config.file_in_directory("embeddings", FILENAME_FAISS_INDEX), idx
        )

        sp.green.ok("✔ ")


def extract_db_properties_en():
    config = Config()
    db = Database()

    sql = """
    SELECT p.id, l.value
    FROM properties p, labels l
    WHERE p.id = l.id
      AND p.value != 'external-id'
      AND l.language = 'en'
    """

    with yaspin(text="Extracting Properties...") as sp:
        property_ids, property_values = [], []
        for id, value in db.fetchall(sql):
            property_ids.append(id)
            property_values.append(value)

        ids_file = config.file_in_directory("embeddings", FILENAME_PROPERTY_IDS)
        _dump_strs_to_file(ids_file, property_ids)

        values_file = config.file_in_directory("embeddings", FILENAME_PROPERTY_LABELS)
        _dump_strs_to_file(values_file, property_values)

        sp.green.ok("✔ ")


def compute_property_embeddings():
    config = Config()

    with yaspin(text="Computing Property Embeddings...") as sp:
        values_file = config.file_in_directory("embeddings", FILENAME_PROPERTY_LABELS)
        values = _read_strs_from_file(values_file)

        embeddings = Transformer().encode(values)
        embeddings_file = config.file_in_directory(
            "embeddings", FILENAME_PROPERTY_EMBEDDINGS
        )
        np.save(embeddings_file, embeddings)

        sp.green.ok("✔ ")


def compute_property_faiss_index():
    config = Config()

    with yaspin(text="Computing Faiss Index...") as sp:
        embeddings = np.load(
            config.file_in_directory("embeddings", FILENAME_PROPERTY_EMBEDDINGS)
        )
        factory_settings = "Flat"
        idx = faiss.index_factory(
            embeddings.shape[1], factory_settings, faiss.METRIC_INNER_PRODUCT
        )
        idx.add(embeddings)
        _dump_pickle_to_file(
            config.file_in_directory("embeddings", FILENAME_PROPERTY_FAISS), idx
        )

        sp.green.ok("✔ ")


def accept(options):
    # Label Embeddings
    if not options.get("skip_extract_labels", False):
        extract_db_labels()
    if not options.get("skip_label_embeddings", False):
        compute_label_embeddings()

    # Alias Embeddings
    if not options.get("skip_extract_aliases", False):
        extract_db_aliases_en()
    if not options.get("skip_aliases_embeddings", False):
        compute_aliases_embeddings()

    # Faiss Index for Labels + Aliases
    if not options.get("skip_faiss", False):
        compute_faiss_index()

    # Property Embeddings
    if not options.get("skip_extract_properties", False):
        extract_db_properties_en()
    if not options.get("skip_property_embeddings", False):
        compute_property_embeddings()

    if not options.get("skip_property_faiss", False):
        compute_property_faiss_index()
