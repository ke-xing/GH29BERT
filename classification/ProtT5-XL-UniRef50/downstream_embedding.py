from tqdm import tqdm
from sklearn.model_selection import KFold
import numpy as np
import torch
from transformers import T5Tokenizer, T5EncoderModel
from Bio import SeqIO
import re
from time import time


from sklearn.model_selection import KFold


def read_fasta(file_path):
    sequences = []
    for record in SeqIO.parse(file_path, "fasta"):
        sequences.append(str(record.seq))

    return sequences


# Load the tokenizer
tokenizer = T5Tokenizer.from_pretrained(
    "Rostlab/prot_t5_xl_half_uniref50-enc", do_lower_case=False, cache_dir="./"
)

# Load the model
model = T5EncoderModel.from_pretrained(
    "Rostlab/prot_t5_xl_half_uniref50-enc", cache_dir="./"
)


#### example usage:


all_x = []
all_y = []
kf = KFold(n_splits=10, shuffle=True, random_state=48)


for i in range(1, 46):
    path_file = f"../../data/new/fasta/cluster_{i}_combined.fasta"

    all_sequence = read_fasta(path_file)
    target = np.zeros(45)
    target[i - 1] = 1.0

    for sequence in all_sequence:
        sequence_examples = [sequence]
        sequence_examples = [
            " ".join(list(re.sub(r"[UZOB]", "X", sequence)))
            for sequence in sequence_examples
        ]
        # tokenize sequences and pad up to the longest sequence in the batch
        ids = tokenizer(sequence_examples, add_special_tokens=True, padding="longest")

        input_ids = torch.tensor(ids["input_ids"]).to("cuda:0")
        attention_mask = torch.tensor(ids["attention_mask"]).to("cuda:0")

        # generate embeddings
        embedding_repr = model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_states = embedding_repr.last_hidden_state.to("cpu")
        sequence_output = last_hidden_states.squeeze()
        sequence_output = sequence_output.detach().numpy()

        all_x.append(sequence_output)
        all_y.append(target)

    print(i)


all_x = np.array(all_x, dtype=object)

i = 0
for train_index, test_index in kf.split(all_x):
    train_x = all_x[train_index]
    train_y = [all_y[i] for i in train_index]
    test_x = all_x[test_index]
    test_y = [all_y[i] for i in test_index]
    i += 1
    np.savez(
        f"../../data/new/npz/train/T5_train_fold{i}.npz",
        x=train_x,
        y=train_y,
        allow_pickle=True,
    )
    np.savez(
        f"../../data/new/npz/test/T5_test_fold{i}.npz",
        x=test_x,
        y=test_y,
        allow_pickle=True,
    )
    break
