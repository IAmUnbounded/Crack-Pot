from __future__ import print_function
import numpy as np
import operator
import torch
import pickle
from fpdf import FPDF
import glob
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from gaps_dataset import gaps
import encoding
class TCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(TCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 96, kernel_size=11, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            encoding.nn.Encoding(D=256,K=32)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(32*256,2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, num_classes),
            nn.Softmax()
        )

    def forward(self, x):
        x = self.features(x)
        #print(x.size())
        x = x.view(x.size(0), 256*32)
        #print (x.size())
        x = self.classifier(x)
        #print (x.size())
        return x

def main():
    train_x = []
    train_y = []
    for i in range(20):
        x_train0, y_train0 = gaps.load_chunk(i, subset='train',datadir='/home/turing/temp')
        for sample in range(x_train0.shape[0]):
            train_x.append(x_train0[sample])
            train_y.append(y_train0[sample])
    train_x = np.array(train_x)
    train_y = np.array(train_y)
    #train_y = np.expand_dims(train_y,1)
    print ("loaded_training data")
    #print (train_x.shape)
    #print (train_y.shape)
    model = TCNN()
    model.cuda()
    optimizer = optim.SGD(model.parameters(), lr=0.00001,momentum=0.9)
    criterion = nn.BCEWithLogitsLoss()
    for epoch in range(40):  # loop over the dataset multiple times
        for i in range(0, train_x.shape[0], 64):
            inputs = []
            labels = []
            for j in range(i,i+128):
                if j < train_x.shape[0]:
                    inputs.append(train_x[j])
                    #print (int(train_y[j]==0))
                    labels.append([int(train_y[j]==0),int(train_y[j]==1)])
            if len(labels) == 128:
                inputs = np.asarray(inputs, dtype=np.float32)
                labels = np.asarray(labels, dtype=np.float32)
                inputs, labels = Variable(torch.from_numpy(inputs).cuda()), Variable(torch.from_numpy(labels).cuda())
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs,labels)
                print("Current loss :")
                print (loss.data[0])
                loss.backward()
                optimizer.step()
        true = 0
        inputs = []
        labels = []
        for k in range(0,32):
            inputs.append(train_x[k])
        inputs = np.asarray(inputs, dtype=np.float32)
        inputs = Variable(torch.from_numpy(inputs).cuda())
        outputs =  model(inputs)
        #print (outputs.size())
        outputs = outputs.max(dim = 1)[1]
        results = outputs.data.cpu().numpy()
        for i in  range(32):
            if(results[k]==train_y[k]):
                true = true + 1
        print ("Correct outputs")
        print (true)
        print ("epoch complete")
        torch.save(model.state_dict(), "/home/turing/temp/alex.pt")
main()