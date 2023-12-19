import dataset
from torch.utils.data import DataLoader
from tape.models import modeling_bert
import torch
import matplotlib.pyplot as plt
import time


start_time = time.time()


config = modeling_bert.ProteinBertConfig(
    num_hidden_layers=5, num_attention_heads=8, hidden_size=400
)
# model=modeling_bert.ProteinBertForMaskedLM(config)
model = torch.load("transformer1500_95p.pt")
train_path = "../../data/Pretrain/train_95p.fasta"
train_data = dataset.MaskedData(train_path)

test_path = "../../data/Pretrain/test_5p.fasta"
test_data = dataset.MaskedData(test_path)

batch_size = 4
num_epoches = 100
lr = 1e-7


train_set = DataLoader(
    train_data, shuffle=True, batch_size=batch_size, collate_fn=train_data.collate_fn
)
test_set = DataLoader(
    test_data, shuffle=False, batch_size=batch_size, collate_fn=test_data.collate_fn
)


optimizer = torch.optim.Adam(model.parameters(), lr=lr)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


model.to(device)
if torch.cuda.device_count() > 1:
    print("Using", torch.cuda.device_count(), "GPUs!")
    model = torch.nn.DataParallel(model, device_ids=[0, 1, 2])


perplexity_train = []
perplexity_test = []
acc_train = []
acc_test = []
loss_train = []
loss_test = []
x = []

for epoch in range(num_epoches):
    model.train()
    x.append(epoch)
    running_loss = 0
    perplexity = 0
    acc = 0

    t1 = time.time()
    for batch_data in train_set:
        batch_masked_token_ids = batch_data["input_ids"]
        batch_input_mask = batch_data["input_mask"]
        batch_labels = batch_data["targets"]

        batch_masked_token_ids = batch_masked_token_ids.to(device)
        batch_input_mask = batch_input_mask.to(device)
        batch_labels = batch_labels.to(device)

        outputs = model(batch_masked_token_ids, batch_input_mask, batch_labels)
        loss = outputs[0][0]
        loss = loss.sum()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        per = outputs[0][1]["perplexity"].sum()
        perplexity += per.item()

        prediction_scores = outputs[1]

        true_indices = batch_labels != -1

        predicted_indices = torch.argmax(prediction_scores, dim=-1)

        correct_predictions = (
            predicted_indices[true_indices] == batch_labels[true_indices]
        ).sum()

        accuracy = correct_predictions.item() / true_indices.sum().item()
        acc += accuracy

    t2 = time.time()
    print(
        f"train epoch {epoch + 1},loss:{running_loss / len(train_set)},perplexity:{perplexity/len(train_set)},time:{t2-t1}s,acc:{acc/len(train_set)*100}%"
    )
    loss_train.append(running_loss / len(train_set))
    perplexity_train.append(perplexity / len(train_set))
    acc_train.append(acc / len(train_set) * 100)

    model.eval()
    running_loss = 0
    perplexity = 0
    acc = 0
    t1 = time.time()
    with torch.no_grad():
        for batch_data in test_set:
            batch_masked_token_ids = batch_data["input_ids"]
            batch_input_mask = batch_data["input_mask"]
            batch_labels = batch_data["targets"]

            batch_masked_token_ids = batch_masked_token_ids.to(device)
            batch_input_mask = batch_input_mask.to(device)
            batch_labels = batch_labels.to(device)

            outputs = model(batch_masked_token_ids, batch_input_mask, batch_labels)
            loss = outputs[0][0]
            loss = loss.sum()
            running_loss += loss.item()

            per = outputs[0][1]["perplexity"].sum()
            perplexity += per.item()

            prediction_scores = outputs[1]

            true_indices = batch_labels != -1

            predicted_indices = torch.argmax(prediction_scores, dim=-1)

            correct_predictions = (
                predicted_indices[true_indices] == batch_labels[true_indices]
            ).sum()

            accuracy = correct_predictions.item() / true_indices.sum().item()
            acc += accuracy

        t2 = time.time()
        print(
            f"test epoch {epoch + 1},loss:{running_loss / len(test_set)},perplexity:{perplexity/len(test_set)},time:{t2 - t1}s,acc:{acc/len(test_set)*100}%\n\n"
        )

        loss_test.append(running_loss / len(test_set))
        perplexity_test.append(perplexity / len(test_set))
        acc_test.append(acc / len(test_set) * 100)

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.plot(x, loss_train, color="r", label="loss_train")
plt.plot(x, loss_test, color="b", label="loss_test")
plt.title("loss")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(x, perplexity_train, color="r", label="perplexity_train")
plt.plot(x, perplexity_test, color="b", label="perplexity_test")

plt.title("perplexity")
plt.xlabel("epoch")
plt.ylabel("perlexity")
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(x, acc_train, color="r", label="acc_train")
plt.plot(x, acc_test, color="b", label="acc_test")

plt.title("acc")
plt.xlabel("epoch")
plt.ylabel("acc")
plt.legend()


plt.tight_layout()
plt.show()
plt.savefig("transformer1500_95p_100.png")


model.to("cpu")
torch.save(model, "transformer1500_95p_100.pt")

end_time = time.time()

delta = end_time - start_time

print(f"train time:{delta/60} min")
