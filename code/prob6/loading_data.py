import torch
import bcolz
import pickle
import numpy as np

def load_word_dic():
    #loading data
    train_label = []
    train_txt = []
    max_length = 0
    with open('../data/train.txt', 'r') as file:
        for line in file:
            line = line.strip().split(' ')
            train_label.append(int(line[0]))
            max_length = max(max_length, len(line[1:]))
            train_txt.append(' '.join(line[1:]))
    
    dev_label = []
    dev_txt = []
    with open('../data/dev.txt', 'r') as file:
        for line in file:
            line = line.strip().split(' ')
            dev_label.append(int(line[0]))
            max_length = max(max_length, len(line[1:]))
            dev_txt.append(' '.join(line[1:]))
    
    test_label = []
    test_txt = []
    with open('../data/test.txt', 'r') as file:
        for line in file:
            line = line.strip().split(' ')
            test_label.append(int(line[0]))
            max_length = max(max_length, len(line[1:]))
            test_txt.append(' '.join(line[1:]))
    
    #getting vocaburary
    word_dic = {}
    for line in train_txt+dev_txt+test_txt:
        line = line.split(' ')
        for word in line:
            if word not in word_dic:
                word_dic[word] = len(word_dic)
    
    train = [train_txt, train_label]
    dev = [dev_txt, dev_label]
    test = [test_txt, test_label]
    
    return word_dic, train, dev, test, max_length

def loading_unlabel():
    word_dic, train, dev, test, max_length = load_word_dic()
    
    #getting text file
    unlabel_txt = []
    with open('../data/unlabelled.txt', 'r') as file:
        for line in file:
            unlabel_txt.append(line)
    
    #forming tensor
    unlabel_data_binary = torch.zeros([len(unlabel_txt), len(word_dic)],device = 'cpu')
    for i in range(len(unlabel_txt)):
        line = unlabel_txt[i].strip().split(' ')
        for word in line:
            try:
                index = word_dic[word]
                unlabel_data_binary[i][index] = 1
            except KeyError:
                pass
    
    unlabel_data_embed = torch.zeros([len(unlabel_txt),max_length+1], dtype=torch.long, device='cpu')
    unlabel_length = torch.zeros(len(unlabel_txt))
    for i in range(len(unlabel_txt)):
        line = unlabel_txt[i].strip().split(' ')
        j = 0
        for word in line:
            try:
                unlabel_data_embed[i, j] = word_dic[word]+1
                j += 1
            except KeyError:
                j += 1
        unlabel_length[i] = j
    
    return unlabel_data_binary, unlabel_data_embed, unlabel_length

def loading_data_binary():    
    word_dic, train, dev, test, max_length = load_word_dic()
    train_txt, train_label = train
    dev_txt, dev_label = dev
    test_txt, test_label = test
    #forming tensor
    train_label = torch.tensor(train_label)
    dev_label = torch.tensor(dev_label)
    test_label = torch.tensor(test_label)
        
    train_data = torch.zeros([len(train_label),len(word_dic)], device='cpu')
    dev_data = torch.zeros([len(dev_label),len(word_dic)], device='cpu')
    test_data = torch.zeros([len(test_label),len(word_dic)], device='cpu')
        
    for i in range(len(train_label)):
        line = train_txt[i].strip().split(' ')
        for word in line:
            index = word_dic[word]
            train_data[i][index] = 1
        
    for i in range(len(dev_label)):
        line = dev_txt[i].strip().split(' ')
        for word in line:
            index = word_dic[word]
            dev_data[i][index] = 1
    
    for i in range(len(test_label)):
        line = test_txt[i].strip().split(' ')
        for word in line:
            index = word_dic[word]
            test_data[i][index] = 1
    
    return  word_dic, train_label, train_data, dev_label, dev_data, test_label, test_data

def loading_data_embedding(style='glove'):
    
    word_dic, train, dev, test, max_length = load_word_dic()
    train_txt, train_label = train
    dev_txt, dev_label = dev
    test_txt, test_label = test
    train_label = torch.tensor(train_label)
    dev_label = torch.tensor(dev_label)
    test_label = torch.tensor(test_label)
        
    train_data = torch.zeros([len(train_label),max_length], dtype=torch.long, device='cpu')
    dev_data = torch.zeros([len(dev_label),max_length],dtype=torch.long, device='cpu')
    test_data = torch.zeros([len(test_label),max_length], dtype=torch.long, device='cpu')
    train_length = torch.zeros(len(train_label))
    dev_length = torch.zeros(len(dev_label))
    test_length = torch.zeros(len(test_label))
    
    if style == 'glove':
        glove = load_data_glov()
        weight_matrix = np.zeros((len(word_dic)+1, 50))
        for i in range(len(train_label)):
            line = train_txt[i].strip().split(' ')
            j = 0
            for word in line:
                train_data[i, j] = word_dic[word]+1
                try: 
                    weight_matrix[word_dic[word]+1] = glove[word]
                except KeyError:
                    weight_matrix[word_dic[word]+1] = np.random.normal(scale=0.6, size=(50))
                j += 1
            train_length[i] = j
            
        for i in range(len(dev_label)):
            line = dev_txt[i].strip().split(' ')
            j = 0
            for word in line:
                dev_data[i, j] = word_dic[word]+1
                try: 
                    weight_matrix[word_dic[word]+1] = glove[word]
                except KeyError:
                    weight_matrix[word_dic[word]+1] = np.random.normal(scale=0.6, size=(50))
                j += 1
            dev_length[i] = j
            
        for i in range(len(test_label)):
            line = test_txt[i].strip().split(' ')
            j = 0
            for word in line:
                test_data[i, j] = word_dic[word]+1
                try: 
                    weight_matrix[word_dic[word]+1] = glove[word]
                except KeyError:
                    weight_matrix[word_dic[word]+1] = np.random.normal(scale=0.6, size=(50))
                j += 1
            test_length[i] = j
            
        weight_matrix = torch.tensor(weight_matrix)
        
        return (word_dic, weight_matrix, train_label, train_data, train_length, 
                dev_label, dev_data, dev_length, 
                test_label, test_data, test_length)
    
    else:
        for i in range(len(train_label)):
            line = train_txt[i].strip().split(' ')
            j = 0
            for word in line:
                train_data[i, j] = word_dic[word]+1
                j += 1
            train_length[i] = j
            
        for i in range(len(dev_label)):
            line = dev_txt[i].strip().split(' ')
            j = 0
            for word in line:
                dev_data[i, j] = word_dic[word]+1
                j += 1
            dev_length[i] = j
            
        for i in range(len(test_label)):
            line = dev_txt[i].strip().split(' ')
            j = 0
            for word in line:
                test_data[i, j] = word_dic[word]+1
                j += 1
            test_length[i] = j
                
        return (word_dic, train_label, train_data, train_length, dev_label, dev_data, dev_length, 
                test_label, test_data, test_length)
    

def load_data_glov():
    glove_path = './glove.6B'
    vectors = bcolz.open(f'{glove_path}/6B.50.dat')[:]
    words = pickle.load(open(f'{glove_path}/6B.50_words.pkl', 'rb'))
    word2idx = pickle.load(open(f'{glove_path}/6B.50_idx.pkl', 'rb'))
    glove = {w: vectors[word2idx[w]] for w in words}
    return glove

def loading_cache():
    words = []
    idx = 0
    word2idx = {}
    vectors = bcolz.carray(np.zeros(1), rootdir=f'{glove_path}/6B.50.dat', mode='w')

    with open(f'{glove_path}/glove.6B.50d.txt', 'rb') as f:
        for l in f:
            line = l.decode().split()
            word = line[0]
            words.append(word)
            word2idx[word] = idx
            idx += 1
            vect = np.array(line[1:]).astype(np.float)
            vectors.append(vect)

    vectors = bcolz.carray(vectors[1:].reshape((400000, 50)), rootdir=f'{glove_path}/6B.50.dat', mode='w')
    vectors.flush()
    pickle.dump(words, open(f'{glove_path}/6B.50_words.pkl', 'wb'))
    pickle.dump(word2idx, open(f'{glove_path}/6B.50_idx.pkl', 'wb'))
