from collections.abc import Callable
import json
import socket
import socketserver


class RequestClient:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    def __enter__(self):
        self.socket.connect((self.host, self.port))
        return self

    def __exit__(self, *_):
        self.close()

    def write(self, data):
        buf = json.dumps(data) + "\n"
        self.socket.sendall(buf.encode("utf-8"))

    def read(self):
        data = self.socket.recv(1024)
        while b"\n" not in data:
            data += self.socket.recv(1024)
        return json.loads(data.decode())

    def __getattr__(self, command):
        def request_handler(**kwargs):
            self.write({"request": command, **kwargs})
            return self.read()

        return request_handler

    def close(self):
        self.socket.close()


class request_handler:
    def __init__(self, fn):
        self.fn = fn

    def __set_name__(self, owner, name):
        if not hasattr(owner, "handlers"):
            owner.handlers = dict()
        owner.handlers[name] = self.fn


class RequestServer(socketserver.StreamRequestHandler):
    handlers: dict[str, Callable]

    def __init__(self, *args, **kwargs):
        self.running = True
        super().__init__(*args, **kwargs)

    def handle(self) -> None:
        try:
            while self.running:
                try:
                    msg = self.read()
                except json.decoder.JSONDecodeError:
                    self.error("Malformed Request.")
                    continue

                if "request" not in msg:
                    self.error("Missing Request Type `request`.")
                    continue

                try:
                    handler = self.handlers[msg["request"]]
                except KeyError:
                    self.error("Invalid Request")
                    continue

                handler(self, msg)
        except ConnectionResetError:
            self.close_connection()

    def error(self, msg):
        self.write({"error": msg})

    def read(self):
        msg = self.rfile.readline()
        return json.loads(msg)

    def write(self, data):
        try:
            buf = json.dumps(data) + "\n"
            self.wfile.write(buf.encode("utf-8"))
            self.wfile.flush()
        except BrokenPipeError:
            self.close_connection()

    def close_connection(self):
        self.running = False

    @classmethod
    def start_server(cls, host: str, port: int, **kwargs) -> None:
        def cls_injected(request, client_address, server):
            cls(**kwargs, request=request, client_address=client_address, server=server)

        class TCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
            address_family = socket.AF_INET
            allow_reuse_address = True

        with TCPServer((host, port), cls_injected) as server:  # type: ignore
            server.serve_forever()
