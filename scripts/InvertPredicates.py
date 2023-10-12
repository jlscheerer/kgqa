import json
from yaspin import yaspin

from kgqa.Config import Config
from kgqa.Database import Database


def _count_claims_by_property():
    db = Database()
    sql = """SELECT property, count(*) FROM claims GROUP BY property"""
    counts = dict()
    for property, count in db.fetchall(sql):
        counts[property] = count
    return counts


def _count_non_invertible_by_property():
    db = Database()
    sql = """
          SELECT property, COUNT(DISTINCT entity_id)
          FROM claims c1
          WHERE EXISTS (
            SELECT * FROM claims c2 WHERE c1.property = c2.property AND c1.entity_id = c2.datavalue_entity
          )
          GROUP BY property;
          """
    counts = dict()
    for property, count in db.fetchall(sql):
        counts[property] = count
    return counts


# TODO(jlscheerer) Rename predicates to properties?
def _compute_invertible_predicates():
    config = Config()
    db = Database()

    with yaspin(text="Counting Claims by Property") as sp:
        pid_to_total = _count_claims_by_property()
        sp.ok("✔ ")

    with yaspin(text="Counting Non-Invertible Claims by Property") as sp:
        pid_to_ninv = _count_non_invertible_by_property()
        sp.ok("✔ ")

    with yaspin(text="Serializing Invertible Properties"):
        predicates = dict()
        for label in db.relations.labels:
            pid = db.relations.label_to_pid[label]

            total = pid_to_total.get(pid, 0)
            invertible = pid_to_ninv.get(pid, 0)

            predicates[pid] = {
                "label": label,
                "total": total,
                "non_invertible": invertible,
            }

        with open(
            config.file_in_directory("predicates", "predicate_info.json"), "w"
        ) as file:
            json.dump(predicates, file)


def accept(options):
    _compute_invertible_predicates()
