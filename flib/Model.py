from flib.utils import convert_size
import torch
import pickle
from torch.utils.data import Dataset, DataLoader


class Model(object):
    def __init__(self, nn: torch.nn.Module) -> None:
        self.nn = nn

    def fit(
        self, dataset: Dataset, loss_fn, optimizer, n_epochs: int, batchsize: int = 64
    ) -> None:
        dataloader = DataLoader(dataset, batch_size=batchsize, shuffle=True)
        print(len(dataset))
        for epoch in range(n_epochs):
            sum_loss = 0
            for batch in dataloader:
                optimizer.zero_grad()
                output = self.nn(batch["X"])
                loss = loss_fn(output, batch["Y"])
                loss.backward()

                sum_loss += loss.item()
                optimizer.step()
                print(
                    "\r[{:3d}/{:3d}] Loss = {:.5f}".format(
                        epoch + 1, n_epochs, sum_loss / len(dataset)
                    ),
                    end="",
                )

    def update(self, weight):
        local_weight = self.nn.state_dict()
        for key in local_weight.keys():
            local_weight[key] -= local_weight[key]
            local_weight[key] += weight[key]

    def weight_dict(self) -> dict:
        w_dict = dict()
        for key in self.nn.state_dict().keys():
            w_dict[key] = self.nn.state_dict()[key]
        return w_dict

    def print_size(self) -> None:
        size = 0
        for key, value in self.weight_dict().items():
            size += len(pickle.dumps(value))
        print("[OK] Weight size:", convert_size(size))

