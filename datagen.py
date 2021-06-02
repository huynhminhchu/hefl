import pickle
import sys
from flib.dataset.spliter import DatasetSpliter

n = int(sys.argv[1])

datapath = "/home/haochu/Desktop/project/datasets/CICIDS2017/custom"
dataset_file_name = "dataset_binary_balance.pickle"  # "dataset_binary_70_30.pickle"
features_train, features_test, labels_train, labels_test = [None] * 4
with open(datapath + "/" + dataset_file_name, "rb") as handle:
    features_train, features_test, labels_train, labels_test = pickle.load(handle)

spliter = DatasetSpliter((features_train, labels_train))
dataset_parts = spliter.split(n)
dataset_parts_name = "dataset_balance_splited_{}_{}.pickle"

for index, part in enumerate(dataset_parts):
    dataset_part_name = datapath + "/" + (dataset_parts_name.format(n, index))
    with open(dataset_part_name, "wb") as handle:
        pickle.dump(
            part, handle, protocol=pickle.HIGHEST_PROTOCOL,
        )
        print(
            "Write dataset with shape {} {} to {}".format(
                part[0].shape, part[1].shape, dataset_part_name
            )
        )
        handle.close()

with open(datapath + "/" + "dataset_test.pickle", "wb") as handle:
    pickle.dump(
        (features_test, labels_test), handle, protocol=pickle.HIGHEST_PROTOCOL,
    )
    print("Save", datapath + "/" + "dataset_test.pickle")
