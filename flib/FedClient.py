from typing import Dict
from flib.network.type import PacketType
from flib.network.packet import (
    ClientHelloPacket,
    InitModelPacket,
    RequestAggregatePacket,
    RequestModelPacket,
    ResponseModelPacket,
    ServerWelcomePacket,
)
from flib.network.base import Client


class FedClient(Client):
    def __init__(self) -> None:
        super().__init__()
        self.__is_leader = False

    def is_leader(self) -> bool:
        return self.__is_leader

    def client_hello(self) -> None:
        client_hello_packet = ClientHelloPacket()
        self.send(client_hello_packet)

        server_welcome_packet = self.recv()
        assert (
            server_welcome_packet.type == PacketType.SERVER_WELCOME_LEADER
            or server_welcome_packet.type == PacketType.SERVER_WELCOME_WORKER
        ), "Packet must be SERVER_WELCOME"

        server_welcome_packet = ServerWelcomePacket.from_packet(server_welcome_packet)
        if server_welcome_packet.is_set_leader():
            print("[OK] Server set this client as leader")
            self.__is_leader = True
        else:
            print("[OK] Server set this client as worker")

    def leader_init_model(self, weight_dict: Dict) -> None:
        init_model_packet = InitModelPacket(weight_dict)
        self.send(init_model_packet)
        print("[OK] Send encrypted init weight to server")

    def request_aggregate(self, weight_dict: Dict, number_of_sample: int) -> None:
        request_aggregate_packet = RequestAggregatePacket(weight_dict, number_of_sample)
        self.send(request_aggregate_packet)
        print("[OK] Send encrypted weight to server for aggregate")

        response_model_packet = self.recv()
        assert (
            response_model_packet.type == PacketType.RESPONSE_MODEL
        ), "Must be RESPONSE_MODEL"
        print("[OK] Recieve model from server")
        response_model_packet = ResponseModelPacket.from_packet(response_model_packet)
        return response_model_packet.get_weight()

    def request_model(self) -> Dict[str, bytes]:
        request_model_packet = RequestModelPacket()
        self.send(request_model_packet)
        print("[OK] Request for init model")

        response_model_packet = self.recv()
        assert (
            response_model_packet.type == PacketType.RESPONSE_MODEL
        ), "Must be RESPONSE_MODEL"
        print("[OK] Recieve model from server")
        response_model_packet = ResponseModelPacket.from_packet(response_model_packet)
        return response_model_packet.get_weight()
