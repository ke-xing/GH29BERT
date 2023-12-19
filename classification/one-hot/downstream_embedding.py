

from sklearn.model_selection import KFold
import numpy as np
import torch
from tape import ProteinBertModel, ProteinBertConfig, TAPETokenizer  # type: ignore
from Bio import SeqIO



def read_fasta(file_path):
    sequences=[]
    for record in SeqIO.parse(file_path,"fasta"):
        sequences.append(str(record.seq))



    return sequences


tokenizer = TAPETokenizer(vocab='iupac')




all_x = []
all_y = []
kf=KFold(n_splits=5,shuffle=True,random_state=48)


for i in range(1,46):
    path_file = f'../../data/new/fasta/cluster_{i}_combined.fasta'

    all_sequence=read_fasta(path_file)
    target=np.zeros(45)
    target[i-1]=1.0





    for sequence in all_sequence:
        token_ids = torch.tensor([tokenizer.encode(sequence)])
        token_ids.squeeze().numpy()
        num_classes = 30

        # 创建单位矩阵
        eye = np.eye(num_classes,dtype=np.double)

        # 获取one-hot编码
        one_hot = eye[token_ids]
        one_hot=np.squeeze(one_hot)

        all_x.append(one_hot)
        all_y.append(target)
    print(i)






all_x=np.array(all_x,dtype=object)

i=0
for train_index,test_index in kf.split(all_x):
    train_x=all_x[train_index]
    train_y=[all_y[i] for i in train_index]
    test_x=all_x[test_index]
    test_y=[all_y[i] for i in test_index]
    i+=1
    np.savez(f'../../data/new/npz/train/train_onehot{i}.npz',x=train_x,y=train_y,allow_pickle=True)
    np.savez(f'../../data/new/npz/test/test_onehot{i}.npz',x=test_x,y=test_y,allow_pickle=True)




