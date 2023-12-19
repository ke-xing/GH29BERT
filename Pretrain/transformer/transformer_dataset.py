
from Bio import SeqIO,SeqRecord
import random


data_path='../../data/Pretrain/pretrain.fasta'



sequences={}

for sequence in SeqIO.parse(data_path,'fasta'):
    sequences[sequence.id]=sequence.seq


keys=list(sequences.keys())
random.shuffle(keys)

shuffled_sequences={key:sequences[key] for key in keys}
train=[]
test=[]

for id in range(int(len(keys)*0.95)):
    seq_record=SeqRecord.SeqRecord(shuffled_sequences[keys[id]],keys[id])
    train.append(seq_record)

for id in range(int(len(keys)*0.95),len(keys)):
    seq_record = SeqRecord.SeqRecord(shuffled_sequences[keys[id]], keys[id])
    test.append(seq_record)


SeqIO.write(train,'../../data/Pretrain/train_95p.fasta','fasta')
SeqIO.write(test,'../../data/Pretrain/test_5p.fasta','fasta')



