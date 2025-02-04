import torch
from torch import nn

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout=0):
        super().__init__()
        self.lstm = nn.LSTM(input_size,hidden_size,num_layers,bias=True,batch_first=True,dropout=dropout)
        self.out = nn.Linear(hidden_size,output_size)
    def forward(self,x):
        x = torch.squeeze(self.lstm(x)[0][:,-1,:])
        return torch.squeeze(self.out(x))
        
class LNN(nn.Module):
    def __init__(self, input_size,output_size,dropout_rate=0):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_size,output_size),
                                 nn.Dropout(dropout_rate))
    def forward(self,x):
        if len(x.shape)>2:
            return self.net(torch.flatten(x,1))
        else: 
            return self.net(x)

class FNN(nn.Module):
    def __init__(self, 
                 input_size,
                 output_size,
                 layers,
                 dropout_rate=0,
                 batch_normalization=False,
                 activ="relu"):
        super().__init__()
        modules = nn.ModuleList()

        modules.append(nn.Linear(input_size,layers[0]))

        if activ=="tanh": modules.append(nn.Tanh())
        if activ=="relu": modules.append(nn.ReLU())
        if activ=="lrelu": modules.append(nn.LeakyReLU())
        if activ=="elu": modules.append(nn.ELU())

        if batch_normalization: modules.append(nn.BatchNorm1d(layers[0]))

        if dropout_rate: modules.append(nn.Dropout(dropout_rate))

        for i in range(len(layers)-1):

            modules.append(nn.Linear(layers[i],layers[i+1]))

            if activ=="tanh": modules.append(nn.Tanh())
            if activ=="relu": modules.append(nn.ReLU())
            if activ=="lrelu": modules.append(nn.LeakyReLU())
            if activ=="elu": modules.append(nn.ELU())

            if batch_normalization: modules.append(nn.BatchNorm1d(layers[i+1]))

            if dropout_rate: modules.append(nn.Dropout(dropout_rate))

        modules.append(nn.Linear(layers[-1],output_size))
        self.net = nn.Sequential(*modules)

    def forward(self,x):
        if len(x.shape)>2:
            return self.net(torch.flatten(x,1))
        else: return self.net(x)

class ElementwiseLinear(nn.Module):
    def __init__(self,input_size,output_size):
        super().__init__()
        self.w = nn.Parameter(torch.rand(input_size),requires_grad=True)
    def forward(self,x):
        if len(x.shape)>2:
            return self.w*(torch.flatten(x,1))
        else: return self.w*x

class ConvolutionalSpectrum(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1,1,5,1,2)
    def forward(self,x):
        return torch.squeeze(self.conv(x.unsqueeze(1)))
