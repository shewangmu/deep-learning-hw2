import time
import torch


def train_embed(traindata, trainlabel, trainlength, batchsize, net, criterion, optimizer, device):
    N, V = traindata.shape
    train_data = traindata.reshape([int(N/batchsize), batchsize, V])
    train_label = trainlabel.reshape([int(N/batchsize), batchsize])
    train_length = trainlength.reshape([int(N/batchsize), batchsize])
    
    history = []
    for epoch in range(10):  # loop over the dataset multiple times
        start = time.time()
        running_loss = 0.0
        for i in range(len(train_label)):
            txt = train_data[i]
            labels = train_label[i]
            length = train_length[i]
            # TODO: zero the parameter gradients
            # TODO: forward pass
            # TODO: backward pass
            # TODO: optimize the network
            optimizer.zero_grad()
            outputs = net(txt, length)
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

def train_bow(traindata, trainlabel, batchsize, net, criterion, optimizer, device):
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


def test_bow(testdata, testlabel, net, device, style):
    correct = 0
    total = len(testlabel)
    with torch.no_grad():
        txt = testdata
        labels = testlabel
        outputs = net(txt)
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the %d %s images: %f %%' % (len(testlabel), style,
        100 * correct / total))

def test_embed(testdata, testlabel, testlength, net, device, style):
    correct = 0
    total = len(testlabel)
    with torch.no_grad():
        txt = testdata
        labels = testlabel
        outputs = net(txt, testlength)
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the %d %s images: %f %%' % (len(testlabel),style,
        100 * correct / total))
    
    
    