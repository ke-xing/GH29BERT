import numpy as np
import matplotlib.pyplot as plt
import pandas
from sklearn.manifold import TSNE
import seaborn as sns
import torch
from tape import TAPETokenizer
import dataset
from matplotlib import cm


from Bio import SeqIO
from torch.utils.data import DataLoader


def read_fasta(file_path):
    sequences = []

    for record in SeqIO.parse(file_path, "fasta"):
        sequences.append(str(record.seq))
    return sequences


df = pandas.read_csv("main_t.csv")
labels = df["cluster"].tolist()
num = [0] * 45
size = []
marker = {}
for se in labels:
    num[se - 1] += 1
total = sum(num)
for se in labels:
    size.append((1 - num[se - 1] / total) * 10)
for i in range(1, 46):
    marker[i] = f"${i}$"


df["size"] = size
plt.figure(dpi=300, figsize=(20, 12))
color = {
    "Actinomycetota": "#008000",
    "Animalia": "#8B4513",
    "Bacillota": "#4169E1",
    "Bacteroidota": "#DC143C",
    "Fungi": "#FF4500",
    "Plantae": "#800080",
    "Pseudomonadota": "#FFD700",
    "others": "#A9A9A9",
}
ax = sns.scatterplot(x="x", y="y", hue="sub", data=df, palette=color, sizes=9)


plt.legend(ncol=9, bbox_to_anchor=(0.85, 0))._fontsize = 30
ax.set_axis_off()
plt.savefig("tsne1.svg")
