import dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
from downstream_model import ClassificationModel2
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import numpy


acc_train = []
acc_test = []
loss_train = []
loss_test = []
x = []


device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")


model = torch.load("class.pt")
optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-6, weight_decay=0.001)
criterion = nn.CrossEntropyLoss()
model.to(device)
train_data_path = f"all_bert.npz"

train_data = dataset.EmbedData(train_data_path)
train_set = DataLoader(train_data, shuffle=False, batch_size=1)

data = []
cl = []
for batch_data in tqdm(train_set):
    input, target = batch_data
    input = input.to(device)
    target = target.to(device)

    optimizer.zero_grad()
    output = model(input).to("cpu")
    cluster = torch.argmax(output, dim=1) + 1

    output = output.squeeze().detach().numpy()
    data.append(output)

    cluster = cluster.numpy()
    cl.append(cluster)

data = numpy.array(data)
cl = numpy.array(cl)

numpy.savez("data.npz", d=data, allow_pickle=True)
numpy.savez("cl.npz", c=cl, allow_pickle=True)
