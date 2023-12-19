import numpy as np
import matplotlib.pyplot as plt
import pandas
from sklearn.manifold import TSNE
import seaborn as sns
import torch
from tape import TAPETokenizer
import dataset
from matplotlib import cm
import numpy
from tqdm import tqdm
from umap import UMAP

from Bio import SeqIO
from torch.utils.data import DataLoader

def read_fasta(file_path):
    sequences=[]

    for record in SeqIO.parse(file_path,"fasta"):
        sequences.append(str(record.name))
    return sequences




label=read_fasta('../data/Pretrain/pretrain.fasta')
data=numpy.load('data.npz')
data=data['d']
cl=numpy.load('cl.npz')
cl=cl['c']


df=pandas.read_csv('main_t-SNE_color.csv')



# 创建UMAP对象并设置参数
reducer = UMAP(n_neighbors=15, min_dist=0.1, n_components=2)
#tsne=TSNE(n_components=2,perplexity=5,random_state=42)
# 拟合数据
data_tsne = reducer.fit_transform(data)
#data_tsne=(tsne.fit_transform(data))





i=0
for x,y in tqdm(data_tsne):
    df.loc[df['name'] == label[i], 'x'] = x
    df.loc[df['name'] == label[i], 'y'] = y
    df.loc[df['name'] == label[i], 'cluster'] = cl[i]
    i+=1

df.to_csv('main_bert_umap.csv')