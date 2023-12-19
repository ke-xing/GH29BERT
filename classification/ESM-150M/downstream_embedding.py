

from sklearn.model_selection import KFold
import numpy as np
import torch
from tape import ProteinBertModel, ProteinBertConfig, TAPETokenizer  # type: ignore
from Bio import SeqIO



from transformers import AutoTokenizer, EsmModel


tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t30_150M_UR50D",cache_dir='./')
model = EsmModel.from_pretrained("facebook/esm2_t30_150M_UR50D",cache_dir='./')


def read_fasta(file_path):
    sequences=[]
    for record in SeqIO.parse(file_path,"fasta"):
        sequences.append(str(record.seq))



    return sequences






#### example usage:



all_x = []
all_y = []
kf=KFold(n_splits=10,shuffle=True,random_state=48)


for i in range(1,46):
    path_file = f'../../data/new/fasta/cluster_{i}_combined.fasta'

    all_sequence=read_fasta(path_file)
    target=np.zeros(45)
    target[i-1]=1.0





    for sequence in all_sequence:
        inputs = tokenizer(sequence, return_tensors="pt")

        outputs = model(**inputs)

        last_hidden_states = outputs.last_hidden_state


        sequence_output=last_hidden_states.squeeze()
        sequence_output=sequence_output.detach().numpy()


        all_x.append(sequence_output)
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
    np.savez(f'../../data/new/npz/train/ESM_150M_train_fold{i}.npz',x=train_x,y=train_y,allow_pickle=True)
    np.savez(f'../../data/new/npz/test/ESM_150M_test_fold{i}.npz',x=test_x,y=test_y,allow_pickle=True)
    break



