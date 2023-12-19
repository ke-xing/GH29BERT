from transformers import AutoTokenizer, EsmModel
import torch

tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D",cache_dir='./')
model = EsmModel.from_pretrained("facebook/esm2_t6_8M_UR50D",cache_dir='./')

print(model)