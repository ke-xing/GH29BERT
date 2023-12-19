
import tqdm
from sklearn.model_selection import KFold
import numpy as np
import torch
from tape import ProteinBertModel, ProteinBertConfig, TAPETokenizer  # type: ignore
from Bio import SeqIO
from tape.models import modeling_bert
import time

def read_fasta(file_path):
    sequences=[]
    for record in SeqIO.parse(file_path,"fasta"):
        sequences.append(str(record.seq))



    return sequences


tokenizer = TAPETokenizer(vocab='iupac')
config=modeling_bert.ProteinBertConfig(num_hidden_layers=5,num_attention_heads=8,hidden_size=400)

model = torch.load('bert.pt').to('cuda:0')


all_x = []
all_y = []




path_file = f'../data/Pretrain/pretrain.fasta'

all_sequence=read_fasta(path_file)
target=np.zeros(45)






for sequence in tqdm.tqdm(all_sequence):
    token_ids = torch.tensor([tokenizer.encode(sequence)])
    token_ids=token_ids.to("cuda:0")
    output=model(token_ids)
    sequence_output=output[1]
    sequence_output=sequence_output.to('cpu')
    sequence_output=sequence_output.squeeze()
    sequence_output=sequence_output.detach().numpy()
    all_x.append(sequence_output)
    all_y.append(target)







all_x=np.array(all_x,dtype=object)

np.savez(f'all_bert.npz',x=all_x,y=all_y,allow_pickle=True)




