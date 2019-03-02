import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from loading_data import *
from solver import *
    
class embedding(nn.Module):
    """
    the structure of embedding class is:
        word embeddings --> average pooling --> Linear --> sigmoid
    reveives the dimension of D of the input words features
    """
    def __init__(self, input_dim=None, word_vec_dim=128, class_num=2, sentence_len = None):
        """
        - input_dim is the length of vocaburary D.
        - word_vec_dim is the dimension of word embeddings W.
        - class_num  is the number of class need to be classified.
        - sentence_len is the dimension of average pooling T.
        """
        super(embedding, self).__init__()
        
        self.class_num = class_num
        self.word_vec_dim = word_vec_dim
        self.input_dim = input_dim
        self.sentence_len = sentence_len
        
        self.embed = nn.Embedding(input_dim, word_vec_dim, padding_idx=0)
        
        
        #self.linear = nn.Linear(sentence_len*word_vec_dim, 2)
        self.linear = nn.Linear(word_vec_dim, 2)
        self.activate = nn.Sigmoid()
    
    def forward(self, x, length):
        embed_word = self.embed(x)
        N = len(length)
        
        out = torch.sum(embed_word, 1)
        
        length_s = length.clone()
        length_s = length_s.unsqueeze_(1)
        length_s = length_s.expand(out.shape)
        out = out/length_s

        
        out = torch.reshape(out, (N, -1))
        out = self.linear(out)
        out = self.activate(out)
        
        return out
    
if __name__ == '__main__':
    word_dic, train_label, train_data, train_length, dev_label, dev_data, dev_length, test_label, test_data, test_length = loading_data_embedding(style='embed')
    index = torch.randperm(len(train_label))
    train_data = train_data[index]
    train_label = train_label[index]

    device = 'cpu'
    net = embedding(input_dim=len(word_dic)+1, word_vec_dim=128, class_num=2, sentence_len = 15).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    batchsize = 100
    history = train_embed(train_data, train_label, train_length, batchsize, net, criterion, optimizer, device)
    
    plt.plot(range(len(history)), history)
    plt.title("learning curve of embedding")
    
    test_embed(train_data, train_label, train_length, net, device, 'train')
    test_embed(dev_data, dev_label, dev_length, net, device, 'dev')
    test_embed(test_data, test_label, test_length, net, device, 'test')
    
    unlabel_data_binary, unlabel_data_embed, unlabel_length = loading_unlabel()
    output = net(unlabel_data_embed, unlabel_length)
    _, predicted = torch.max(output.data, 1)
    with open('./output/predictions_q2.txt', 'w') as txt_file:
        for i in predicted:
            txt_file.write(str(i.item()))
            txt_file.write('\n')
    
    
    