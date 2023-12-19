import torch
import torch.nn as nn
from tape.models import modeling_bert


config=modeling_bert.ProteinBertConfig(hidden_size=30)
class ClassificationModel1(torch.nn.Module):
    def __init__(self):
        super(ClassificationModel1, self).__init__()
        self.dense_layer1 = nn.Linear(1024, 2048)
        self.dense_layer2 = nn.Linear(2048, 512)
        self.output_layer = nn.Linear(512, 45)
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.3)

    def forward(self, protein_sequence):
        hidden_layer_output1 = torch.relu(self.dense_layer1(protein_sequence))
        #hidden_layer_output1 = self.dropout1(hidden_layer_output1)

        hidden_layer_output2 = torch.relu(self.dense_layer2(hidden_layer_output1))
        #hidden_layer_output2 = self.dropout2(hidden_layer_output2)

        output = self.output_layer(hidden_layer_output2)

        return output




class ClassificationModel2(nn.Module):
    def __init__(self):
        super(ClassificationModel2,self).__init__()
        self.attention_layer=nn.Linear(30,1)
        self.hidden_layer=nn.Linear(30,1024)
        self.output_layer=nn.Linear(1024,45)
        self.relu=nn.ReLU()
        self.dropout=nn.Dropout(0.1)
        self.layernorm1=nn.LayerNorm(30)
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