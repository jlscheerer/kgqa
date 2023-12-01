import argparse

from kgqa.Config import Config
from kgqa.FaissIndex import FaissIndexDirectory
from kgqa.FaissServer import FaissServer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="faiss_server")
    parser.add_argument("-d", "--debug", action="store_true")
    args = parser.parse_args()

    if args.debug:
        print("Enabling debug mode for FaissServer")
        FaissServer.debug_mode = True
    else:
        _ = FaissIndexDirectory()
    print("FaissServer successfully loaded Index Directory.")

    config = Config()["embeddings"]["server"]
    FaissServer.start_server(config["host"], config["port"])
