from kgqa.Config import Config
from kgqa.FaissIndex import FaissIndexDirectory
from kgqa.FaissServer import FaissServer

if __name__ == "__main__":
    _ = FaissIndexDirectory()
    print("FaissServer successfully loaded Index Directory.")

    config = Config()["embeddings"]["server"]
    FaissServer.start_server(config["host"], config["port"])
