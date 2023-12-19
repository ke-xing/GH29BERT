import dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
from downstream_model import ClassificationModel2
import matplotlib.pyplot as plt
import time


acc_train = []
acc_test = []
loss_train = []
loss_test = []
x = []


num_epoches = 50

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


model = ClassificationModel2()
optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-6, weight_decay=0.001)
criterion = nn.CrossEntropyLoss()
model.to(device)
train_data_path = f"../../data/new/npz/train/T5_train_fold1.npz"
test_data_path = f"../../data/new/npz/test/T5_test_fold1.npz"

train_data = dataset.EmbedData(train_data_path)
train_set = DataLoader(train_data, shuffle=True, batch_size=1)

test_data = dataset.EmbedData(test_data_path)
test_set = DataLoader(test_data, shuffle=False, batch_size=1)

for epoch in range(num_epoches):
    x.append(epoch)
    model.train()
    running_loss = 0
    acc = 0
    t1 = time.time()
    for batch_data in train_set:
        input, target = batch_data
        input = input.to(device)
        target = target.to(device)

        optimizer.zero_grad()
        output = model(input)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        output = torch.argmax(output, dim=1)
        target = torch.argmax(target, dim=1)
        correct_predictions = output == target
        correct_predictions = correct_predictions.sum() / len(input)
        acc += correct_predictions.item()

    t2 = time.time()

    print(
        f"train epoch {epoch+1},loss:{running_loss/len(train_set)},acc:{acc/len(train_set)*100},time:{t2-t1}"
    )
    loss_train.append(running_loss / len(train_set))
    acc_train.append(acc / len(train_set) * 100)

    model.eval()
    t1 = time.time()
    with torch.no_grad():
        acc = 0
        running_loss = 0
        perplexity = 0
        for batch_data in test_set:
            input, target = batch_data
            input = input.to(device)
            target = target.to(device)

            output = model(input)
            loss = criterion(output, target)
            running_loss += loss.item()
            per = 2 ** loss.item()
            perplexity += per
            output = torch.argmax(output, dim=1)
            target = torch.argmax(target, dim=1)
            correct_predictions = output == target
            correct_predictions = correct_predictions.sum() / len(input)
            acc += correct_predictions.item()

        t2 = time.time()
        print(
            f"test epoch {epoch + 1},loss:{running_loss / len(test_set)},acc:{acc/len(test_set)*100},time:{t2-t1},perplexity:{perplexity/len(test_set)}\n"
        )
        loss_test.append(running_loss / len(test_set))
        acc_test.append(acc / len(test_set) * 100)


plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(x, loss_train, color="b", label="loss_train")
plt.plot(x, loss_test, color="r", label="loss_test")
plt.title("loss")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(x, acc_train, color="b", label="acc_train")
plt.plot(x, acc_test, color="r", label="acc_test")
plt.title("acc")
plt.xlabel("epoch")
plt.ylabel("acc")
plt.legend()


plt.tight_layout()
plt.show()

plt.savefig(f"fold1.png")

model.to("cpu")
torch.save(model, f"down_model_kfold1.pt")
