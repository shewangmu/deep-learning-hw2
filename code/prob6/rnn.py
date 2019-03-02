from loading_data import *
from torch.autograd import Variable
from solver import *
import torch
import torch.nn as nn
import time
import matplotlib.pyplot as plt

class rnn_class(nn.Module):
    """
    the structure of embedding class is:
        word embeddings --> average pooling --> Linear --> sigmoid
    reveives the dimension of D of the input words features
    """
    def __init__(self, num_layers=1, input_dim=None, word_vec_dim=50, 
                 class_num=2, sentence_len = None, weight_matrix=None,
                 style = 'RNN'):
        """
        - input_dim is the length of vocaburary D.
        - word_vec_dim is the dimension of word embeddings W.
        - class_num  is the number of class need to be classified.
        - sentence_len is the dimension of average pooling T.
        """
        super(rnn_class, self).__init__()
        
        self.class_num = class_num
        self.word_vec_dim = word_vec_dim
        self.input_dim = input_dim
        self.sentence_len = sentence_len
        self.num_layers = num_layers
        
        self.embed = nn.Embedding(input_dim, word_vec_dim, padding_idx=0)
        self.embed.load_state_dict({'weight': weight_matrix})
        
        self.hidden_dim = 200
        self.style = style
        
        if style == 'RNN':
            self.rnn = nn.RNN(input_size=word_vec_dim, hidden_size=self.hidden_dim, 
                          num_layers=self.num_layers, batch_first=True)
        else:
            self.rnn = nn.LSTM(input_size=word_vec_dim, hidden_size=self.hidden_dim, 
                          num_layers=self.num_layers, batch_first=True)
        
        self.linear = nn.Linear(self.hidden_dim, 2)
        self.activate = nn.Sigmoid()
    
    def forward(self, x):
        #embed
        N, T = x.shape
        embed_word = self.embed(x)
        
        if self.style=='RNN':
            #rnn
            _, hn = self.rnn(embed_word, self.init(N)[0])
        else:
            _, hn = self.rnn(embed_word, self.init(N))
            hn, cn = hn
        
        #linear
        ht = hn.view(N, -1)
        out = self.linear(ht)
        
        #sigmoid
        out = self.activate(out)
        
        return out
    
    def init(self, batchsize):
        return (Variable(torch.zeros(self.num_layers, batchsize, self.hidden_dim))
    ,Variable(torch.zeros(self.num_layers, batchsize, self.hidden_dim)))
    
def train(traindata, trainlabel, batchsize, net, criterion, optimizer, device):
    N, V = traindata.shape
    train_data = traindata.reshape([int(N/batchsize), batchsize, V])
    train_label = trainlabel.reshape([int(N/batchsize), batchsize])
    
    history = []
    for epoch in range(10):  # loop over the dataset multiple times
        start = time.time()
        running_loss = 0.0
        for i in range(len(train_label)):
            txt = train_data[i]
            labels = train_label[i]
            # TODO: zero the parameter gradients
            # TODO: forward pass
            # TODO: backward pass
            # TODO: optimize the network
            optimizer.zero_grad()
            outputs = net(txt)
            #outputs = net(txt, length, mask)
            running_loss = criterion(outputs, labels)
            history.append(running_loss)
            running_loss.backward()
            optimizer.step()
            
            # print statistics
            # running_loss += loss.item()
            if i % 100 == 99:    # print every 2000 mini-batches
                end = time.time()
                print('[epoch %d, iter %5d] loss: %.3f eplased time %.3f' %
                      (epoch + 1, i + 1, running_loss / 100, end-start))
                start = time.time()
                running_loss = 0.0
    print('Finished Training')
    return history

def test(testdata, testlabel, net, device, batchsize):
    correct = 0
    total = len(testlabel)
    N, V = testdata.shape
    test_data = testdata.reshape([int(N/batchsize), batchsize, V])
    test_label = testlabel.reshape([int(N/batchsize), batchsize])
    with torch.no_grad():
        for i in range(len(test_label)):
            txt = test_data[i]
            labels = test_label[i]
            outputs = net(txt)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
        print('Accuracy of the network on the %d test images: %f %%' % (len(testlabel),
            100 * correct / total))

    
if __name__ == '__main__':
    word_dic, weight_matrix, train_label, train_data, train_length, dev_label, dev_data, dev_length, test_label, test_data, test_length = loading_data_embedding(style='glove')
    index = torch.randperm(len(train_label))
    train_data = train_data[index]
    train_label = train_label[index]
    
    device = 'cpu'
    batchsize = 100
    style='RNN'
    net = rnn_class(num_layers=1, input_dim=len(word_dic)+1, word_vec_dim=50, class_num=2,
                    sentence_len=15, weight_matrix=weight_matrix, style=style).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    history = train_bow(train_data, train_label,  batchsize, net, criterion, optimizer, device)
    
    
    plt.plot(range(len(history)), history)
    plt.title("learning curve of {}".format(style))
    
    test_bow(train_data, train_label,  net, device, 'train')
    test_bow(dev_data, dev_label,  net, device, 'dev')
    test_bow(test_data, test_label,  net, device, 'test')
    
    unlabel_data_binary, unlabel_data_embed, unlabel_length = loading_unlabel()
    output = net(unlabel_data_embed)
    _, predicted = torch.max(output.data, 1)
    with open('./output/predictions_q4.txt', 'w') as txt_file:
        for i in predicted:
            txt_file.write(str(i.item()))
            txt_file.write('\n')
        
        
        