import psycopg2
import numpy as np

from kgqa.FileUtils import read_strs_from_file

from .Singleton import Singleton
from .Config import Config
from .Constants import (
    FILENAME_PROPERTY_EMBEDDINGS,
    FILENAME_PROPERTY_IDS,
    FILENAME_PROPERTY_LABELS,
)


class Relations:
    def __init__(self, labels, embeddings, ids):
        self.labels = labels
        self.embeddings = embeddings

        self.label_to_pid = dict()
        self.pid_to_label = dict()
        for label, pid in zip(labels, ids):
            self.label_to_pid[label] = pid
            self.pid_to_label[pid] = label


class Database(metaclass=Singleton):
    def __init__(self):
        config = Config()
        self._conn = psycopg2.connect(
            f"dbname={config['psql']['db']} user={config['psql']['user']} password={config['psql']['pwd']}"
        )

        # TODO(jlscheerer) We can directly move this into the relations
        labels = read_strs_from_file(
            config.file_in_directory("embeddings", FILENAME_PROPERTY_LABELS)
        )
        embeddings = np.load(
            config.file_in_directory("embeddings", FILENAME_PROPERTY_EMBEDDINGS)
        )
        ids = read_strs_from_file(
            config.file_in_directory("embeddings", FILENAME_PROPERTY_IDS)
        )
        self.relations = Relations(labels, embeddings, ids)

    def cursor(self):
        return self._conn.cursor()

    def execute(self, sql):
        cursor = self.cursor()
        cursor.execute(sql)

    def commit(self):
        self._conn.commit()

    def fetchall(self, sql):
        cursor = self.cursor()
        cursor.execute(sql)
        return cursor.fetchall()

    def fetchmany(self, sql, batch_size):
        cursor = self.cursor()
        cursor.execute(sql)
        cursor_row = cursor.fetchmany(batch_size)
        while cursor_row:
            yield cursor_row
            cursor_row = cursor.fetchmany(batch_size)

    def get_qid_to_titles(self, qids):
        raise AssertionError("NYI")

    def get_qid_to_title(self, qid):
        raise AssertionError("NYI")

    def get_pid_to_titles(self, pids):
        raise AssertionError("NYI")

    def get_pid_to_title(self, pid):
        raise AssertionError("NYI")

    def __del__(self):
        if self._conn is not None:
            self._conn.close()
