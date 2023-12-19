
from tqdm import tqdm
from sklearn.model_selection import KFold
import numpy as np
import torch
from transformers import T5Tokenizer, T5EncoderModel
from Bio import SeqIO
import re
from time import time
from transformers import AutoTokenizer, EsmModel

def read_fasta(file_path):
    sequences=[]
    for record in SeqIO.parse(file_path,"fasta"):
        sequences.append(str(record.seq))

    return sequences

# Load the tokenizer

tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t30_150M_UR50D", cache_dir='/')
model = EsmModel.from_pretrained("facebook/esm2_t30_150M_UR50D", cache_dir='/')


#### example usage:



all_x = []
all_y = []




path_file = f'../../data/Pretrain/pretrain.fasta'

all_sequence=read_fasta(path_file)
target=np.zeros(45)

model=model.to('cuda:0')



for sequence in tqdm(all_sequence):


    inputs = tokenizer(sequence, return_tensors="pt").to('cuda:0')

    outputs = model(**inputs)

    last_hidden_states = outputs.last_hidden_state

    sequence_output = last_hidden_states.squeeze().to('cpu')
    sequence_output = sequence_output.detach().numpy()

    torch.cuda.empty_cache()

    all_x.append(sequence_output)
    all_y.append(target)







all_x=np.array(all_x,dtype=object)


np.savez(f'all.npz',x=all_x,y=all_y,allow_pickle=True)




