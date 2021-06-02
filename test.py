from lstm import INPUT_SIZE
import pickle
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

dataset_train_path = (
    "/home/haochu/Desktop/project/datasets/CICIDS2017/custom/dataset_splited_4_0.pickle"
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

HIDDEN_LAYER = 32
INPUT_SIZE = 78
OUTPUT_SIZE = 15


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


class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.features[index], self.labels[index]


train_dataset = CustomDataset(features_train, labels_train)
test_dataset = CustomDataset(features_test, labels_test)
trainloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
testloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import confusion_matrix, accuracy_score

epochs = 10
batch_size = 64
loss = 0

model = LSTMMultiple()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()
loss_vals = []
acc_vals = []
score_vals = []

for epoch in range(epochs):
    running_loss = 0.0
    for step, (inputs, labels) in enumerate(trainloader):
        inputs = inputs.view(-1, 1, INPUT_SIZE).float()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        running_loss += loss.item()
        loss.backward()
        optimizer.step()

    loss_vals.append(running_loss)

    sum = 0
    accuracy = 0
    pred_list = []
    for step, (a_x, a_y) in enumerate(testloader):
        a_x = a_x.view(-1, 1, INPUT_SIZE).float()
        test_output = model(a_x)
        pred_y = torch.max(test_output, 1)[1]
        pred_list.extend(pred_y)
        for i in range(len(a_y)):
            if pred_y[i] == a_y[i]:
                sum = sum + 1

    accuracy = float(sum) / float(len(test_dataset))
    print(
        "Epoch: ",
        epoch,
        "| train loss: %.4f" % running_loss,
        "| test accuracy: %.4f" % accuracy,
    )
    acc_vals.append(accuracy)
    # print(score(labels_test, pred_list))
    precision, recall, fscore, support = score(labels_test, pred_list)
    precision = precision * 100
    recall = recall * 100
    score_vals.append(score)
    for i in range(OUTPUT_SIZE):
        print(
            "Label: ",
            i,
            "| precision: %.2f" % precision[i],
            "%",
            "| recall: %.2f" % recall[i],
            "%",
            "| F1: %.2f" % fscore[i],
            "| support:",
            support[i],
        )

