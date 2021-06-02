from __future__ import annotations
import pickle
from typing import Dict

import tenseal as ts
from flib.network.type import PacketType
from flib.network.constant import *
from struct import pack, unpack


class Packet(object):
    def __init__(self) -> None:
        self.content = b""
        self.type = PacketType.NONE

    def __init__(self, content: bytes, type: PacketType) -> None:
        self.content = content
        self.type = type

    def serialize(self) -> bytes:
        self.full = (
            pack(
                PACKET_LENGTH_FORMAT,
                len(self.content)
                + NUMBER_OF_PACKET_TYPE_BYTE
                + NUMBER_OF_PACKET_LENGTH_BYTE,
            )
            + pack(PACKET_TYPE_FORMAT, self.type.value)
            + self.content
        )
        return self.full

    @staticmethod
    def from_bytes(data: bytes) -> Packet:
        type = unpack(
            PACKET_TYPE_FORMAT,
            data[
                NUMBER_OF_PACKET_LENGTH_BYTE : NUMBER_OF_PACKET_LENGTH_BYTE
                + NUMBER_OF_PACKET_TYPE_BYTE
            ],
        )[0]
        content = data[NUMBER_OF_PACKET_LENGTH_BYTE + NUMBER_OF_PACKET_TYPE_BYTE :]
        return Packet(content=content, type=type)

    def content_length(self):
        return len(self.content)

    def packet_length(self):
        return (
            self.content_length()
            + NUMBER_OF_PACKET_LENGTH_BYTE
            + NUMBER_OF_PACKET_TYPE_BYTE
        )


class ClientHelloPacket(Packet):
    def __init__(self) -> None:
        super().__init__(CLIENT_HELLO_MSG, PacketType.CLIENT_HELLO)


class ServerWelcomePacket(Packet):
    def __init__(self, is_leader=False) -> None:
        self.__is_leader = is_leader
        if is_leader:
            super().__init__(SERVER_WELCOME_MSG, PacketType.SERVER_WELCOME_LEADER)
        else:
            super().__init__(SERVER_WELCOME_MSG, PacketType.SERVER_WELCOME_WORKER)

    @staticmethod
    def from_packet(packet: Packet) -> ServerWelcomePacket:
        if packet.type == PacketType.SERVER_WELCOME_WORKER:
            return ServerWelcomePacket()
        return ServerWelcomePacket(is_leader=True)

    def is_set_leader(self):
        return self.__is_leader


class InitModelPacket(Packet):
    def __init__(self, weight: Dict, is_bytes=False) -> None:
        if not is_bytes:
            weight = pickle.dumps(weight)
        super().__init__(weight, PacketType.INIT_MODEL)

    @staticmethod
    def from_packet(packet: Packet) -> InitModelPacket:
        return InitModelPacket(packet.content, is_bytes=True)

    def get_weight(self) -> Dict[str, bytes]:
        weight = pickle.loads(self.content)
        return weight


class RequestModelPacket(Packet):
    def __init__(self) -> None:
        super().__init__(REQUEST_MODEL_MSG, PacketType.REQUEST_MODEL)


class ResponseModelPacket(Packet):
    def __init__(self, weight: Dict, is_bytes=False) -> None:
        if not is_bytes:
            weight = pickle.dumps(weight)
        super().__init__(weight, PacketType.RESPONSE_MODEL)

    @staticmethod
    def from_packet(packet: Packet):
        return ResponseModelPacket(packet.content, is_bytes=True)

    def get_weight(self) -> Dict[str, bytes]:
        weight = pickle.loads(self.content)
        return weight


class RequestAggregatePacket(Packet):
    def __init__(self, weight: Dict, n_sample: int = 0, is_bytes=False) -> None:
        if not is_bytes:
            weight = pickle.dumps(weight)
            self.n_sample = n_sample
        else:
            self.n_sample = unpack(
                PACKET_LENGTH_FORMAT, weight[:NUMBER_OF_PACKET_LENGTH_BYTE]
            )[0]
            weight = weight[NUMBER_OF_PACKET_LENGTH_BYTE:]
        n_sample = pack(PACKET_LENGTH_FORMAT, self.n_sample)
        super().__init__(n_sample + weight, PacketType.REQUEST_AGGREGATE)

    @staticmethod
    def from_packet(packet: Packet) -> RequestAggregatePacket:
        return RequestAggregatePacket(packet.content, is_bytes=True)

    def get_weight(self) -> Dict[str, bytes]:
        weight = pickle.loads(self.content[NUMBER_OF_PACKET_LENGTH_BYTE:])
        return weight

    def get_number_sample(self) -> int:
        return self.n_sample
