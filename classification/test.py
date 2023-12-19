import torch
import numpy as np
from Bio import SeqIO
from tape import TAPETokenizer
import dataset
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
def read_fasta(file_path):
    sequences=[]

    for record in SeqIO.parse(file_path,"fasta"):
        sequences.append(str(record.seq))
    return sequences

pretrained_model=torch.load('../Pretrain/trans/transformer1500.pt')

downstream_model=torch.load('down_model_kfold1.pt')




sequences=read_fasta(f'TTenzymes.fasta')

target = np.zeros(45)
tokenizer = TAPETokenizer(vocab='iupac')
test_x=[]
test_y=[]

for sequence in sequences:
    token_ids = torch.tensor([tokenizer.encode(sequence)])
    output = pretrained_model(token_ids)
    sequence_output = output[1]
    sequence_output = sequence_output.squeeze()
    sequence_output = sequence_output.detach().numpy()

    test_x.append(sequence_output)
    test_y.append(target)

test_x=np.array(test_x,dtype=object)
np.savez(f'test.npz',x=test_x,y=test_y,allow_pickle=True)


test_data=dataset.EmbedData(f'test.npz')
test_data=DataLoader(test_data,batch_size=1,shuffle=False)


with torch.no_grad():
    for sequence in test_data:
        input, target = sequence
        output = downstream_model(input)
        output = torch.softmax(output, dim=1)
        entropy = -torch.sum(output * torch.log2(output))
        print(f'cluster:{torch.argmax(output, dim=1).item() + 1},entropy:{entropy}')