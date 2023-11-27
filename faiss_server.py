from kgqa.RequestServer import RequestServer, request_handler
from kgqa.FaissIndex import FaissIndexDirectory
from kgqa.MatchingUtils import compute_similar_entity_ids, compute_similar_predicates


class FaissServer(RequestServer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @request_handler
    def compute_similar_properties(self, data):
        predicate = data["property"]
        # TODO(jlscheerer) Rename the function in MatchingUtils.
        pids, scores = compute_similar_predicates(predicate)
        self.write({"pids": pids, "scores": [float(x) for x in scores]})

    @request_handler
    def compute_similar_entities(self, data):
        entity = data["entity"]
        # TODO(jlscheerer) Rename the function in MatchingUtils and clean it up.
        qids, scores = compute_similar_entity_ids(entity)
        self.write({"qids": qids, "scores": [float(x) for x in scores]})


if __name__ == "__main__":
    _ = FaissIndexDirectory()
    print("FaissServer successfully loaded Index Directory.")
    FaissServer.start_server("127.0.01", 43096)
