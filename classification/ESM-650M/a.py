from transformers import AutoTokenizer, EsmModel
import torch

tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D",cache_dir='./')
model = EsmModel.from_pretrained("facebook/esm2_t33_650M_UR50D",cache_dir='./')

inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
outputs = model(**inputs)

last_hidden_states = outputs.last_hidden_state

print(last_hidden_states.shape)