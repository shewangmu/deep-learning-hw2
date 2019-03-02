# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from solver import *


class WordEmbeddingRNN(nn.Module):
    def __init__(self, num_labels, num_embeddings, hidden_size, num_layers=1, embedding_dim=50, weights_matrix=None, use_pretrain=False):
        super(WordEmbeddingRNN,self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim, padding_idx=0)
        if use_pretrain:
            self.embedding.load_state_dict({'weight': torch.tensor(weights_matrix)})
        self.rnn = nn.RNN(self.embedding_dim, self.hidden_size, self.num_layers, batch_first=True)
        self.linear1 = nn.Linear(self.hidden_size, num_labels)
        
    def forward(self, x):
        N = x.shape[0]
        x = self.embedding(x)
        _, h = self.rnn(x, self.init_hidden(N))
#        h = torch.zeros(self.num_layers, N, self.hidden_size)
#        for n in range(N):
#            h[0, n, :] = h_t[n, int(lengths[n]-1), :]
#        h = Variable(h)
        return F.softmax(self.linear1(h.view(N,-1)))

    def init_hidden(self, batch_size):
        return Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size)) 

#def main():
#vocab_idx = create_vocab('./data')
#filepath = './glove.6B/glove.6B.50d.txt'
#glove = LoadGloVe(filepath)
#trainset = LoadDataset('./data/train.txt', vocab_idx, batch_size=100)
#weights_matrix = create_weights(vocab_idx, glove, embed_dim=50)
#weight_matrix, train_label, train_data, train_length, dev_label, dev_data, dev_length = loading_data_embedding(style='glove')
index = torch.randperm(len(train_label))
train_data = train_data[index]
train_label = train_label[index]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = WordEmbeddingRNN(2, 8095, hidden_size=200, weights_matrix=weight_matrix, use_pretrain=False).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)
#
history = train_bow(train_data, train_label, batchsize, net, criterion, optimizer, device)
#
#print('Train.txt result:')
#test(trainset, model, device)
#
#devset = LoadDataset('./data/dev.txt', vocab_idx, batch_size=100)
#print('Dev.txt result:')
test_bow(dev_data, dev_label, net, device)
#
#testset = LoadDataset('./data/test.txt', vocab_idx, batch_size=100)
#print('Test.txt result:')
#test(testset, model, device)

#if __name__ == '__main__':
#    main()