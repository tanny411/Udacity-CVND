import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)
        
    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1, drop_prob=0.0):
        super().__init__()
        self.embed_size=embed_size
        self.hidden_size=hidden_size
        self.vocab_size=vocab_size
        self.num_layers=num_layers
        self.drop_prob=drop_prob
        
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, dropout=drop_prob, batch_first=True)
        self.linear_fc = nn.Linear(hidden_size, vocab_size)
#         self.lstm_out = nn.LSTM(hidden_size, vocab_size, num_layers, dropout=drop_prob, batch_first=True)
    
    def forward(self, features, captions):
        
        # Discard the <end> word
        captions = captions[:, :-1] 
        
        x = self.embedding(captions)
        features = features.unsqueeze(dim=1)
        x = torch.cat((features,x),dim=1)
        x,_ = self.lstm(x)
        x = self.linear_fc(x)
        return x

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        res = []
        hidden = (torch.zeros(1, 1, self.hidden_size),torch.zeros(1, 1, self.hidden_size))
        for i in range(max_len):
            outputs, hidden = self.lstm(inputs, hidden)
            outputs = self.linear_fc(outputs.squeeze(1))
            target_index = outputs.max(1)[1] ##select the index
            res.append(target_index.item())
            inputs = self.embedding(target_index).unsqueeze(1)
        return res