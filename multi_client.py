from flib.dataset.spliter import DatasetSpliter
import pickle
import tenseal as ts
from sys import argv

import torch
import numpy as np
from flib.crypto.seal import SEAL
from flib.FedClient import FedClient
from lstm import CustomModel, LSTMMultiple

# Get HE key
context = None
with open("keys/seal.cxt", "rb") as handle:
    context = ts.context_from(handle.read())
    print("[OK] Load SEAL context")
    handle.close()
assert context != None, "SEAl context must not be None"
seal = SEAL(context)

n_client = int(argv[1])
client_id = int(argv[2])
n_round = int(argv[3])

# Get dataset
dataset_train_path = "/home/haochu/Desktop/project/datasets/CICIDS2017/custom/dataset_balance_splited_{}_{}.pickle".format(
    n_client, client_id
)
print("Load dataset from {}", dataset_train_path)

dataset_test_path = (
    "/home/haochu/Desktop/project/datasets/CICIDS2017/custom/dataset_test.pickle"
)

features_train, labels_train = None, None
features_test, labels_test = None, None

with open(dataset_train_path, "rb") as handle:
    features_train, labels_train = pickle.load(handle)
    handle.close()

with open(dataset_test_path, "rb") as handle:
    features_test, labels_test = pickle.load(handle)
    handle.close()

# Create model
net = LSTMMultiple()
local_model = CustomModel(net)
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
print("[OK] Init model")

client = FedClient()
client.connect()
client.client_hello()

if client.is_leader():
    enc_dict = seal.encrypt(local_model.weight_dict())
    enc_ser_dict = seal.serialize(enc_dict)
    client.leader_init_model(enc_ser_dict)
else:
    init_weight = client.request_model()
    init_weight = seal.deserialize(init_weight)
    init_weight = seal.decrypt(init_weight)
    local_model.update(init_weight)

spliter = DatasetSpliter((features_train, labels_train))
dataset_parts = spliter.split(n_round)

# features_train, labels_train = dataset_parts[0]

for round, (features_train, labels_train) in enumerate(dataset_parts):
    # if round > 0:
    #     features_train = np.concatenate((features_train, features_train_part))
    #     labels_train = np.concatenate((labels_train, labels_train_part))
    # features_train = features_train_part
    # labels_train = labels_train_part

    print("Round {}:".format(round + 1))
    print("Train with {} samples".format(len(features_train)))
    local_model.fit(
        features_train,
        labels_train,
        # features_test,
        # labels_test,
        criterion=loss_fn,
        optimizer=optimizer,
        n_epochs=10,
        batchsize=64,
    )
    # print("")
    # print(local_model.weight_dict()["out.bias"])
    # if round == n_round:
    print("Local test")
    local_model.test(features_test, labels_test)  # Local test

    enc_dict = seal.encrypt(local_model.weight_dict())
    enc_ser_dict = seal.serialize(enc_dict)
    updated_weight = client.request_aggregate(enc_ser_dict, len(features_train))
    updated_weight = seal.deserialize(updated_weight)
    updated_weight = seal.decrypt(updated_weight)
    local_model.update(updated_weight)

    # print(local_model.weight_dict()["out.bias"])
    if client.is_leader() and round == n_round - 1:
        print("Final test")
        local_model.test(features_test, labels_test)
