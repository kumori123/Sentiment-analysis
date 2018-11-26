# !/usr/bin/python
# coding: utf8
import gensim
import matplotlib.pyplot as plt
import numpy as np
import time
import torch
import torch.nn.utils.rnn as rnn
from torch import nn, optim

from torch.autograd import Variable

from torch.utils.data import DataLoader

from torchvision import datasets, transforms
import torch.utils.data as data


import net

dir = r'C:\Users\37581\Downloads\all\\'
xs = []
ys = []
testxs = []
testys = []
senlen = 40
sen_len=[]

# Load Google's pre-trained Word2Vec model.
start = time.time()
wmodel = gensim.models.KeyedVectors.load_word2vec_format(
    r'C:\Users\37581\Downloads\GoogleNews-vectors-negative300.bin\GoogleNews-vectors-negative300.bin',
    binary=True)
end1 = time.time()
print(end1 - start)
# print(model['apple'])

with open(dir + 'training.txt', encoding='utf-8') as f:
    for line in f.readlines():
        tmp = line.split('\t')
        ys.append(int(tmp[0]))
        tmp2 = tmp[1].split(' ')
        x = [[0]*300]*senlen
        count = 0
        for word in tmp2:
            if word.isdigit():
                continue
            elif '.\n' in word:
                word = word[:-2]
            if word in wmodel:
                x[count] = wmodel[word]
            else:
                x[count] = [0]*300
        count = count + 1
        xs.append(x)
    sen_len.append(count)

with open(dir + 'testdata1.txt', encoding='utf-8') as f:
    for line in f.readlines():
        tmp = line.split('\t')
        testys.append(int(tmp[0]))
        tmp2 = tmp[1].split(' ')
        sen_vec = [[0]*300]*senlen
        count = 0
        for word in tmp2:
            if word.isdigit():
                continue
            elif '.\n' in word:
                word = word[:-2]
            if word in wmodel:
                sen_vec[count] = wmodel[word]
            else:
                sen_vec[count] = [0]*300
            count = count + 1
        testxs.append(sen_vec)

xs = np.array(xs)
ys = np.array(ys, dtype=np.int64)
testxs = np.array(testxs)
testys = np.array(testys)

xs = torch.Tensor(xs)
ys = torch.Tensor(ys)
testxs = torch.Tensor(testxs)
testys = torch.Tensor(testys)

print(xs.shape, x[0].shape, ys.shape)
print(testxs.shape, testxs[0].shape, testys.shape)

end2 = time.time()
print(end2-start)


class MyDataset(data.Dataset):
    def __init__(self, x, y):
        self.tx = x
        self.ty = y

    def __getitem__(self, index):
        ttx, tty = self.tx[index], self.ty[index]
        return ttx, tty

    def __len__(self):
        return len(self.ty)


dataset = MyDataset(xs, ys)
testdataset = MyDataset(testxs, testys)


# 定义一些超参数

num_epochs = 5
embed_size = 300
num_hiddens = 100
num_layers = 2
bidirectional = False
batch_size = 200
labels = 2
lr = 0.01




# 数据集的下载器

train_dataset = dataset
test_dataset = testdataset

#test_dataset = datasets.MNIST(root='./data', train=False, transform=data_tf)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# 选择模型

#model = net.simpleNet(28 * 28, 300, 100, 10)

# model = net.Activation_Net(28 * 28, 300, 100, 10)

model = net.SentimentNet(embed_size,num_hiddens,num_layers,bidirectional,labels)
model.train()
#if torch.cuda.is_available():

#    model = model.cuda()


# 定义损失函数和优化器

criterion = nn.CrossEntropyLoss()

#optimizer = optim.SGD(model.parameters(), lr=learning_rate)
optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.99))
#optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, alpha = 0.9)


# 训练模型

epoch = 0
epoch_arr=[]
loss_arr=[]
for data in train_loader:

    feature, label = data
    #print(img.shape, label.shape)
    label.long()
#    img = img.view(img.size(0), -1)

    if torch.cuda.is_available():

        feature = feature.cuda()

        label = label.cuda()

    else:

        feature = Variable(feature)

        label = Variable(label)

    #print(label.type)
    label = label.long()
    #print(label)
    out = model(feature)

    loss = criterion(out, label)

    print_loss = loss.data.item()
    loss_arr.append(print_loss)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    epoch += 1
    epoch_arr.append(epoch)
    if epoch % num_epochs == 0:

        print('epoch: {}, loss: {:.4}'.format(epoch, loss.data.item()))
plt.figure()
plt.plot(epoch_arr, loss_arr)
plt.xlabel('epoches')
plt.ylabel('trainning loss')
plt.show()
print('start testing...............')
# 模型评估

model.eval()

eval_loss = 0

eval_acc = 0

for data in test_loader:

    feature, label = data

   # img = img.view(img.size(0), -1)
    label = label.long()
    if torch.cuda.is_available():

        feature = feature.cuda()

        label = label.cuda()

    out = model(feature)

    loss = criterion(out, label)

    eval_loss += loss.data.item() * label.size(0)

    _, pred = torch.max(out, 1)

    num_c = (pred == label)
    num_correct = (pred == label).sum()
    pt = pred.sum().item()
    rt = label.sum().item()
    tp = (num_c.long() * pred.long()).sum().item()

    eval_acc += num_correct.item()

print('Test Loss: {:.6f}, Acc: {:.6f}'.format(

    eval_loss / (len(test_dataset)),

    eval_acc / (len(test_dataset))
))
print('precsion:{:.6f} \n'
      'recall: {:.6f}'.format(tp/pt,tp/rt))
print('end---------------------')
