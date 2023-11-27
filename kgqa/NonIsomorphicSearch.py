from kgqa.MatchingUtils import compute_similar_entities
from kgqa.PromptBuilder import PromptBuilder
from kgqa.Database import Database


def pids_to_sql_list(pids):
    return ", ".join([f"'{x}'" for x in pids])


def infer_n_hops_from_lists(subj_pids, obj_pids, n: int = 1, inverse: bool = False):
    db = Database()
    basetable = "claims_5m_inv"
    subj_list = pids_to_sql_list(subj_pids)
    obj_list = pids_to_sql_list(obj_pids)
    SELECT = ", ".join([f"c{i + 1}.property" for i in range(n)])
    FROM = ", ".join([f"{basetable} c{i + 1}" for i in range(n)])
    WHERE = " AND ".join(
        [f"c1.entity_id in ({subj_list})", f"c{n}.datavalue_entity IN ({obj_list})"]
        + [f"c{i}.datavalue_entity = c{i + 1}.entity_id" for i in range(1, n)]
    )
    return db.fetchall(f"SELECT {SELECT} FROM {FROM} WHERE {WHERE}")


def infer_n_hops_from_examples(examples, n: int = 1):
    options = []
    for subj, obj in examples:
        subj_pids, subj_scores = compute_similar_entities(subj, num_qids=5)
        obj_pids, obj_scores = compute_similar_entities(obj, num_qids=5)
        options.append(set(infer_n_hops_from_lists(subj_pids, obj_pids, n=n)))

    properties = set.intersection(*options)
    return properties


def infer_n_hops_predicate(predicate, n: int = 1):
    db = Database()
    examples = (
        PromptBuilder(template="PROMINENT_EXAMPLES")
        .set("RELATIONSHIP", predicate)
        .execute()
    )
    print("LLM - Examples:", examples)
    results = infer_n_hops_from_examples(examples, n)

    print(f"{n}-hops: ")
    for entry in results:
        print(" ".join([f"'{db.get_pid_to_title(pid)}'" for pid in entry]))
