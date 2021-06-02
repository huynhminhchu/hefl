import pickle
import tenseal as ts
from sys import argv

import torch
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

# Get dataset
dataset_train_path = "/home/haochu/Desktop/project/datasets/CICIDS2017/custom/dataset_splited_6_{}.pickle".format(
    argv[1]
)

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

local_model.fit(
    features_train,
    labels_train,
    features_test,
    labels_test,
    criterion=loss_fn,
    optimizer=optimizer,
    n_epochs=2,
    batchsize=64,
)
print("")
print(local_model.weight_dict()["out.bias"])
local_model.test(features_test, labels_test)

enc_dict = seal.encrypt(local_model.weight_dict())
enc_ser_dict = seal.serialize(enc_dict)
updated_weight = client.request_aggregate(enc_ser_dict, len(features_test))
updated_weight = seal.deserialize(updated_weight)
updated_weight = seal.decrypt(updated_weight)
local_model.update(updated_weight)

print(local_model.weight_dict()["out.bias"])
if client.is_leader():
    local_model.test(features_test, labels_test)
