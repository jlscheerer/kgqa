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
            non_invertible = pid_to_ninv.get(pid, 0)

            predicates[pid] = {
                "label": label,
                "total": total,
                "non_invertible": non_invertible,
            }

        with open(
            config.file_in_directory("predicates", "predicate_info.json"), "w"
        ) as file:
            json.dump(predicates, file)

        sp.ok("✔ ")


def _construct_inv_predicate_table():
    config = Config()
    db = Database()

    with yaspin(text="Determining Invertible Predicates...") as sp:
        with open(
            config.file_in_directory("predicates", "predicate_info.json"), "r"
        ) as file:
            predicates = json.load(file)

        NON_INVERTIBLE_REL_THRESHOLD = config["predicates"]["non_invertible"][
            "rel_threshold"
        ]
        NON_INVERTIBLE_ABS_THRESHOLD = config["predicates"]["non_invertible"][
            "abs_threshold"
        ]

        non_invertible_predicates, invertible_predicates = [], []
        for pid, data in predicates.items():
            total = data["total"]
            non_invertible = data["non_invertible"]
            if (
                non_invertible > NON_INVERTIBLE_REL_THRESHOLD * total
                or non_invertible > NON_INVERTIBLE_ABS_THRESHOLD
            ):
                non_invertible_predicates.append(pid)
            else:
                invertible_predicates.append(pid)

        sp.ok("✔ ")

    print(f"Counting {len(invertible_predicates)} / {len(predicates)} as invertible.")

    with yaspin(text="Populating claims_inv Table (Inverted Predicates)...") as sp:
        escaped_invertible_predicates = [f"'{pid}'" for pid in invertible_predicates]
        sql = f"""
        INSERT INTO claims_inv
        SELECT datavalue_entity AS entity_id, id, property, datavalue_type, datavalue_string, 
               entity_id AS datavalue_entity, datavalue_date, datavalue_quantity, True AS inverse
        FROM claims_inv
        WHERE datavalue_type = 'wikibase-entityid'
          AND property IN ({', '.join(escaped_invertible_predicates)});
        """
        db.execute(sql)
        sp.ok("✔ ")

    db.commit()


def _construct_inv_predicate_idx():
    db = Database()

    with yaspin(text="Constructing Indexes for claims_inv Table...") as sp:
        db.execute("CREATE INDEX idx_claims_inv_entity_id ON claims_inv (entity_id);")
        db.execute(
            "CREATE INDEX idx_claims_inv_datavalue_entity ON claims_inv (datavalue_entity);"
        )
        db.execute("CREATE INDEX idx_claims_inv_property ON claims_inv (property);")
        db.execute(
            "CREATE INDEX idx_claims_inv_entity_id_property ON claims_inv (entity_id, property);"
        )
        db.execute(
            "CREATE INDEX idx_claims_inv_property_datavalue_entity ON claims_inv (property, datavalue_entity);"
        )

        db.commit()

        sp.ok("✔ ")


def accept(options):
    if not options.get("skip_count_predicates", False):
        _compute_invertible_predicates()

    if not options.get("skip_create_table", False):
        _construct_inv_predicate_table()

    _construct_inv_predicate_idx()
