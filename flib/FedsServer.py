import socket
import threading
from typing import Dict
import tenseal as ts

from flib.crypto.seal import SEAL
from flib.network.constant import DEFAULT_ADDRESS
from flib.network.packet import (
    InitModelPacket,
    RequestAggregatePacket,
    ResponseModelPacket,
    ServerWelcomePacket,
)
from flib.network.type import PacketType
from flib.network.base import Server


class FedServer(Server):
    def __init__(self, address: tuple = DEFAULT_ADDRESS, n_client=2) -> None:
        super().__init__(address=address)
        self.number_client_required = n_client

        self.leader = None
        self.weight: Dict = None
        self.weight_list: list[Dict] = []
        self.number_sample_list = []
        self.lock = threading.Lock()

        # Get HE key
        context = None
        with open("keys/seal.pub", "rb") as handle:
            context = ts.context_from(handle.read())
            print("[OK] Load SEAL context")
            handle.close()
        assert context != None, "SEAl context must not be None"
        self.seal = SEAL(context)

    def FedAvg(self):
        sum_sample = sum(self.number_sample_list)
        self.weight = self.weight_list[0]
        for key, value in self.weight.items():
            self.weight[key] = value * (self.number_sample_list[0] / sum_sample)

        for index, weight in enumerate(self.weight_list):
            if index == 0:
                continue
            for key, value in weight.items():
                self.weight[key] += value * (
                    self.number_sample_list[index] / sum_sample
                )

    def handle(self, client: socket.socket) -> None:
        packet = self.recv(client)
        # CLIENT_HELLO PACKET
        if packet.type == PacketType.CLIENT_HELLO:
            self.lock.acquire()
            if not self.leader:
                self.leader = client
                server_welcome_leader = ServerWelcomePacket(is_leader=True)
                self.send(client, server_welcome_leader)

                init_model_packet = self.recv(client)
                assert (
                    init_model_packet.type == PacketType.INIT_MODEL
                ), "Must be INIT_MODEL packet"
                init_model_packet = InitModelPacket.from_packet(init_model_packet)
                self.weight = init_model_packet.get_weight()
                self.weight = self.seal.deserialize(self.weight)
                assert len(self.weight) > 0, "Weight size must be greater than 0"
                print("[OK] Recieve init model from leader")
            self.lock.release()
            if client is not self.leader:
                server_welcome_worker = ServerWelcomePacket()
                self.send(client, server_welcome_worker)
        # REUQEST_MODEL_PACKET
        elif packet.type == PacketType.REQUEST_MODEL:
            print("[OK] Recieve request model from client")
            weight_ser = self.seal.serialize(self.weight)
            response_model_packet = ResponseModelPacket(weight_ser)
            self.send(client, response_model_packet)
            print("[OK] Send model to client")
        elif packet.type == PacketType.REQUEST_AGGREGATE:
            request_aggregate_packet = RequestAggregatePacket.from_packet(packet)
            weight_dict = request_aggregate_packet.get_weight()
            number_of_sample = request_aggregate_packet.get_number_sample()
            weight_dict = self.seal.deserialize(weight_dict)
            assert len(self.weight) > 0, "Weight size must be greater than 0"
            print("[OK] Recieve aggregate request from client")

            self.lock.acquire()
            self.weight_list.append(weight_dict)
            self.number_sample_list.append(number_of_sample)

            if len(self.weight_list) == self.number_client_required:
                self.FedAvg()
                print("Aggregated, boardcast")
                weight_ser = self.seal.serialize(self.weight)
                response_model_packet = ResponseModelPacket(weight_ser)
                for s in self.clients:
                    self.send(s, response_model_packet)
                self.weight_list.clear()
                self.number_sample_list.clear()
            self.lock.release()

