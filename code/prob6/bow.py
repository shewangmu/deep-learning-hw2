import torch
import torch.nn as nn
from torch.autograd import Variable
import time
import matplotlib.pyplot as plt
from loading_data import *
from solver import *

class bow(nn.Module):
    """
    bow produces the way of text classification by the net: 
        bag of words --> linear --> sigmoid
    receives the dimension D of input words features.
    """
    def __init__(self, input_dim=None, class_num=2):
        """
        construct classfification instance.
        inputs:
        - input_dim is the length of vocaburary D.
        - class_num  is the number of class needed to be classified.
        """
        super(bow, self).__init__()
        
        self.class_num = class_num
        
        self.linear = nn.Linear(input_dim, class_num)
        self.activate = nn.Sigmoid()
        
    def forward(self, bow):
        x = self.linear(bow)
        out = self.activate(x)
        #out = F.log_softmax(out)
        return out
 
if __name__ == '__main__':
    device = 'cpu'
    word_dic, train_label, train_data, dev_label, dev_data, test_label, test_data = loading_data_binary()
    index = torch.randperm(len(train_label))
    train_data = train_data[index]
    train_label = train_label[index]
    
    net = bow(input_dim=len(word_dic)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    batchsize = 100
        
            
    history = train_bow(train_data, train_label, batchsize, net, criterion, optimizer, device)
         
    test_bow(train_data, train_label, net, device, 'train')
    test_bow(dev_data, dev_label, net, device, 'dev')
    test_bow(test_data, test_label, net, device, 'test')
        
    plt.plot(range(len(history)), history)  
    plt.title("learning curve of bow")
    
    
    unlabel_data_binary, unlabel_data_embed, unlabel_length = loading_unlabel()
    output = net(unlabel_data_binary)
    _, predicted = torch.max(output.data, 1)
    with open('./output/predictions_q1.txt', 'w') as txt_file:
        for i in predicted:
            txt_file.write(str(i.item()))
            txt_file.write('\n')
    
    