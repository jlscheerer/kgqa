from kgqa.FaissIndex import FaissIndexDirectory

# TODO(jlscheerer) Reimplment the LanguageModel follow up logic and alias handling.
from .Config import Config


class Edge:
    def __init__(self, user_predicate):
        self.up = user_predicate
        self.pids = []

    def set_pids(self, pids):
        self.pids = pids

    def get_up(self):
        return self.up

    def get_pids(self):
        return self.pids

    def __repr__(self):
        return self.up


# TODO(jlscheerer) Reimplment alias handling logic here.
def compute_similar_predicates(pred_english):
    """
    For a given predicate specified in english, returns a list of predicates,
    their ids, and assoc. probs in the form:
        [prob:int, pid:str, label:str]
    where each probability is the cosine similarity (inner product) with the given predicate.

    Returns top-num_preds such entities.
    """
    config = Config()
    pid_to_score = FaissIndexDirectory().properties.search(
        pred_english, config["NumTopPID"]
    )
    pids_scores = sorted(
        [(pid, score) for pid, score in pid_to_score.items()],
        key=lambda x: x[1],
        reverse=True,
    )
    pids, scores = zip(*pids_scores)
    return pids, scores


def match_predicates(aqg):
    """
    Updates aqg.edges s.t. aqg.edges[(i,j)] = List[pid]
    """
    pred_to_pid_to_score = dict()
    for k, v in aqg.edges.items():
        if v.get_up() == "instance of":
            pids, scores = ["P31"], [1.0]
        else:
            pids, scores = compute_similar_predicates(v.get_up())
        scores = [x / 2 for x in scores]
        pred_to_pid_to_score[v.get_up()] = dict(zip(pids, scores))
        aqg.edges[k].set_pids(pids)
        # TODO(jlscheerer) Reimplment follow up via LanguageModel here.
    return pred_to_pid_to_score


# TODO(jlscheerer) Reimplment alias handling logic here.
def compute_similar_entity_ids(e_english):
    """
    For a given entity specified in english, returns a list of lists of the follwing form:
        [prob:int, qid:str, label:str, id:int]
    where each probability is the cosine similarity (inner product) with the given entity.

    Returns top-num_qids such entities.
    """
    config = Config()
    num_qids = config["NumTopQID"]
    pid_to_score = FaissIndexDirectory().labels.search(e_english, num_qids)
    pids_scores = sorted(
        [(pid, score) for pid, score in pid_to_score.items()],
        key=lambda x: x[1],
        reverse=True,
    )
    pids, scores = zip(*pids_scores)

    return pids, scores


def match_entities(aqg):
    """
    Updates aqg.anchors_to_wiki
    """
    ent_to_qid_to_score = dict()
    for node in aqg.nodes:
        if not node.is_free:
            ent_string = node.value
            qids, scores = compute_similar_entity_ids(ent_string)
            # TODO(jlscheerer) Reimplment follow up via LanguageModel here.
            scores = [round(x, 2) for x in scores]
            aqg.anchors_to_wiki[node.id] = qids
            ent_to_qid_to_score[ent_string] = dict(zip(qids, scores))

    return ent_to_qid_to_score
