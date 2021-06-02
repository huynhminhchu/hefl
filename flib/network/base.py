import socket
import struct
import threading

from numpy import select
from flib.network.constant import *
from flib.network.packet import Packet


class Server(object):
    def __init__(self, address: tuple = DEFAULT_ADDRESS) -> None:
        super().__init__()
        # Create socket
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind(address)
        self.socket.listen(5)

        # List of client socket
        self.clients: list[socket.socket] = []
        print("[OK] Server is runining...")

    def send(self, client: socket.socket, packet: Packet) -> None:
        client.send(packet.serialize())

    def recv(self, client: socket.socket) -> Packet:
        bytes_recive = bytearray()
        header_recive = b""

        # Recieve header first
        while not header_recive:
            header_recive = client.recv(
                NUMBER_OF_PACKET_LENGTH_BYTE + NUMBER_OF_PACKET_TYPE_BYTE
            )
        bytes_recive.extend(header_recive)

        total_packet_length = struct.unpack(
            PACKET_LENGTH_FORMAT, header_recive[:NUMBER_OF_PACKET_LENGTH_BYTE]
        )[0]

        while len(bytes_recive) != total_packet_length:
            buf = min(total_packet_length - len(bytes_recive), STANDARD_BUFFER_SIZE)
            bytes_recive.extend(client.recv(buf))

        return Packet.from_bytes(bytes(bytes_recive))

    def accept(self) -> socket.socket:
        client, _ = self.socket.accept()
        self.clients.append(client)
        print("[OK] New client connected")
        return client

    def wait(self) -> None:
        while True:
            super().__init__()
            try:
                client = self.accept()
                thread = threading.Thread(
                    target=self.handle_loop, args=(client,), daemon=True
                )
                thread.start()
            except KeyboardInterrupt:
                for i in self.clients:
                    i.close()
                self.socket.close()
                break

    def handle_loop(self, client):
        while True:
            self.handle(client)

    def handle(self, client: socket.socket) -> None:
        pass


class Client(object):
    def __init__(self) -> None:
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    def connect(self, address: tuple = DEFAULT_ADDRESS) -> None:
        self.socket.connect(address)
        print("[OK] Connected to server")

    def send(self, packet: Packet) -> None:
        data_bytes = packet.serialize()
        self.socket.send(data_bytes)

    def recv(self) -> Packet:
        bytes_recive = bytearray()
        header_recive = b""

        # Recieve header first
        while not header_recive:
            header_recive = self.socket.recv(
                NUMBER_OF_PACKET_LENGTH_BYTE + NUMBER_OF_PACKET_TYPE_BYTE
            )
        bytes_recive.extend(header_recive)

        total_packet_length = struct.unpack(
            PACKET_LENGTH_FORMAT, header_recive[:NUMBER_OF_PACKET_LENGTH_BYTE]
        )[0]

        while len(bytes_recive) != total_packet_length:
            buf = min(total_packet_length - len(bytes_recive), STANDARD_BUFFER_SIZE)
            bytes_recive.extend(self.socket.recv(buf))

        return Packet.from_bytes(bytes(bytes_recive))
