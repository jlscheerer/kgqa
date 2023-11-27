from kgqa.FaissIndex import FaissIndexDirectory
from kgqa.FaissServer import FaissServer

if __name__ == "__main__":
    _ = FaissIndexDirectory()
    print("FaissServer successfully loaded Index Directory.")
    FaissServer.start_server("127.0.01", 43096)
