from flib.crypto.seal import SEAL
from flib.FedClient import FedClient
from flib.Model import Model
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from sklearn.metrics import precision_recall_fscore_support as score

torch.manual_seed(1)
INPUT_SIZE = 78
BATCH_SIZE = 64
OUTPUT_SIZE = 2
HIDDEN_LAYER = 32


class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.features[index], self.labels[index]


class LSTMMultiple(nn.Module):
    def __init__(self):
        super(LSTMMultiple, self).__init__()
        self.lstm = nn.LSTM(
            input_size=INPUT_SIZE,
            hidden_size=HIDDEN_LAYER,
            num_layers=1,
            batch_first=True,
        )
        self.out = nn.Linear(HIDDEN_LAYER, OUTPUT_SIZE)

    def forward(self, x):
        r_out, (h_n, h_c) = self.lstm(x, None)
        return self.out(r_out[:, -1, :])


class CustomModel(Model):
    def test(self, features_test, labels_test, batchsize: int = 64):
        testset = CustomDataset(features_test, labels_test)
        testloader = DataLoader(testset, batch_size=batchsize, shuffle=False)
        sum = 0
        accuracy = 0
        pred_list = []
        for inputs, labels in testloader:
            inputs = inputs.view(-1, 1, INPUT_SIZE).float()
            test_output = self.nn(inputs)
            pred_y = torch.max(test_output, 1)[1]
            pred_list.extend(pred_y)
            for i in range(len(labels)):
                if pred_y[i] == labels[i]:
                    sum = sum + 1

        accuracy = float(sum) / float(len(testset))
        print("Accuracy: %.4f" % accuracy,)
        precision, recall, fscore, support = score(labels_test, pred_list)
        for i in range(OUTPUT_SIZE):
            print(
                "Label: ",
                i,
                "| precision: %.4f" % precision[i],
                "%",
                "| recall: %.4f" % recall[i],
                "%",
                "| F1: %.4f" % fscore[i],
                "| support:",
                support[i],
            )

    def fit(
        self,
        features_train,
        labels_train,
        # features_test,
        # labels_test,
        criterion,
        optimizer,
        n_epochs: int,
        batchsize: int,
    ) -> None:
        train_data = CustomDataset(features_train, labels_train)
        # test_data = CustomDataset(features_test, labels_test)
        trainloader = DataLoader(train_data, batch_size=batchsize, shuffle=True)
        # testloader = DataLoader(test_data, batch_size=batchsize, shuffle=False)

        for epoch in range(n_epochs):
            running_loss = 0.0
            for inputs, labels in trainloader:
                inputs = inputs.view(-1, 1, INPUT_SIZE).float()
                outputs = self.nn(inputs)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                running_loss += loss.item()
                loss.backward()
                optimizer.step()

            print(
                "Epoch: ", epoch, "| train loss: %.4f" % running_loss,
            )
