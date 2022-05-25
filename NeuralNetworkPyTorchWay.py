# -*- coding:utf-8 -*-

import math
import numpy as np
import torch
from torch import nn
import torchvision.datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

ComputeDevice = "cuda" if torch.cuda.is_available() else "cpu"

training_data = torchvision.datasets.FashionMNIST(
	root="data",
	train=True,
	download=False,
	transform=torchvision.transforms.ToTensor(),
)

test_data = torchvision.datasets.FashionMNIST(
	root="data",
	train=False,
	download=False,
	transform=torchvision.transforms.ToTensor(),
)

batch_size = 64

train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

i_historyList = []
J_historyList = []


class NeuralNetworkModel(torch.nn.Module):
	def __init__(self):
		super(NeuralNetworkModel, self).__init__()

		self.LayerFunc = torch.nn.Linear(1, 1, device=ComputeDevice)
		self.flatten = nn.Flatten()
		self.linear_relu_stack = nn.Sequential(
			nn.Linear(28 * 28, 512),
			nn.ReLU(),
			nn.Linear(512, 512),
			nn.ReLU(),
			nn.Linear(512, 10)
		)

	def forward(self, x: torch.tensor) -> torch.tensor:
		x = self.flatten(x)
		logits = self.linear_relu_stack(x)
		return logits


def getLoss(y_pred, y):
	LossFunc = torch.nn.CrossEntropyLoss()
	J = LossFunc(y_pred, y)
	return J


def train(dataloader, model, optimizer, iternum):
	for epoch in range(iternum):
		for batch, (X, y) in enumerate(dataloader):
			X, y = X.to(ComputeDevice), y.to(ComputeDevice)
			pred = model(X)
			loss = getLoss(pred, y)

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			J_historyList.append(loss.detach().item())
			i_historyList.append(epoch)


def test(dataloader, model):
	size = len(dataloader.dataset)
	num_batches = len(dataloader)
	model.eval()
	test_loss, correct = 0, 0
	with torch.no_grad():
		for X, y in dataloader:
			X, y = X.to(ComputeDevice), y.to(ComputeDevice)
			pred = model(X)
			test_loss += getLoss(pred, y).item()
			correct += (pred.argmax(1) == y).type(torch.float).sum().item()
	test_loss /= num_batches
	correct /= size
	print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


Model = NeuralNetworkModel()
Model.to(device=ComputeDevice)

Optimizer = torch.optim.SGD(Model.parameters(), lr=1e-3)

train(train_dataloader, Model, Optimizer, 10)

torch.save(Model.state_dict(), r".\Model.pth")

plt.figure(1)
plt.plot(i_historyList, J_historyList)
plt.show()
