import torch
import torch.nn as nn
from tape.models import modeling_bert


config=modeling_bert.ProteinBertConfig(hidden_size=480)

class ClassificationModel2(nn.Module):
    def __init__(self):
        super(ClassificationModel2,self).__init__()
        self.attention_layer=nn.Linear(480,1)
        self.hidden_layer=nn.Linear(480,1024)
        self.output_layer=nn.Linear(1024,45)
        self.relu=nn.ReLU()
        self.dropout=nn.Dropout(0.1)
        self.layernorm1=nn.LayerNorm(480)
        self.layernorm2=nn.LayerNorm(1024)
        self.attention_selfoutput= modeling_bert.ProteinBertSelfOutput(config)

    def forward(self,sequence):
        sequence=self.layernorm1(sequence)
        attention_values=self.attention_layer(sequence)
        attention_values=self.dropout(attention_values)
        attention_weights=torch.softmax(attention_values,dim=1)



        weighted_embeddings=sequence*attention_weights
        attention_embeddings=torch.sum(weighted_embeddings,dim=1)

        attention_embeddings=self.attention_selfoutput(attention_embeddings,attention_embeddings)
        attention_embeddings=self.dropout(attention_embeddings)
        attention_embeddings = self.attention_selfoutput(attention_embeddings, attention_embeddings)

        hidden_output=self.dropout(self.relu(self.hidden_layer(attention_embeddings)))
        hidden_output=self.layernorm2(hidden_output)
        output=self.output_layer(hidden_output)

        return output