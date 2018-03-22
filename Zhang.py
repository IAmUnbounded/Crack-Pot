import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.init as init
import torch.utils.model_zoo as model_zoo
from gaps_dataset import gaps
import encoding
import torch.optim as optim
import glob
from torch.autograd import Variable
class Fire(nn.Module):

    def __init__(self, inplanes, squeeze_planes,
                 expand1x1_planes, expand3x3_planes):
        super(Fire, self).__init__()
        self.inplanes = inplanes
        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes,
                                   kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes,
                                   kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat([
            self.expand1x1_activation(self.expand1x1(x)),
            self.expand3x3_activation(self.expand3x3(x))
        ], 1)


class SqueezeNet(nn.Module):

    def __init__(self, version=1.0, num_classes=2):
        super(SqueezeNet, self).__init__()
        if version not in [1.0, 1.1]:
            raise ValueError("Unsupported SqueezeNet version {version}:"
                             "1.0 or 1.1 expected".format(version=version))
        self.num_classes = num_classes
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            Fire(64, 16, 64, 64),
            Fire(128, 16, 64, 64),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            Fire(128, 32, 128, 128),
            Fire(256, 32, 128, 128),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            Fire(256, 48, 192, 192),
            Fire(384, 48, 192, 192),
            Fire(384, 64, 256, 256),
            Fire(512, 64, 256, 256),
        )
        self.layer = encoding.nn.Encoding(D=512,K=32) 
        self.classifier = nn.Sequential(
            nn.Dropout(p = 0.5),
            nn.Linear(32*512,num_classes),
            nn.ReLU(inplace=True),
            nn.Softmax()
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_uniform(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        x = self.layer(x)
        x = x.view(x.size(0), 512*32)
        x = self.classifier(x)
        
        print x.size()
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
    model = SqueezeNet()
    model.cuda()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    criterion = nn.MSE()
    for epoch in range(40):  # loop over the dataset multiple times
        for i in range(0, train_x.shape[0], 32):
            inputs = []
            labels = []
            for j in range(i,i+32):
                if j < train_x.shape[0]:
                    inputs.append(train_x[j])
                    #print (int(train_y[j]==0))
                    if(train_y[j]==0):
                        v1 = 0
                        v2 = 1
                    else:
                        v1 = 1
                        v2 = 0   
                    labels.append([v1,v2])
            if len(labels) == 32:
                inputs = np.asarray(inputs, dtype=np.float32)
                labels = np.asarray(labels, dtype=np.float32)
                optimizer.zero_grad()
                inputs, labels = Variable(torch.from_numpy(inputs).cuda()), Variable(torch.from_numpy(labels).cuda())
                outputs = model(inputs)
                loss = criterion(outputs,labels)
                print("Current loss :")
                print (loss.data[0])
                loss.backward()
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
        torch.save(model.state_dict(), "/home/turing/temp/model_encoding.pt")
main()