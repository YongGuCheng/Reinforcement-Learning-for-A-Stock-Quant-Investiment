import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.distributions import Normal
torch.manual_seed(666)
device = torch.device("cpu")     


def Mish(x):
    x = x * (torch.tanh(F.softplus(x)))
    return x



class LSTM_SAC_Actor(nn.Module):
    def __init__(self,state_dim, time_steps=10,
                 action_dim=1,
                 hidden_list=[128,64]):
        super(LSTM_SAC_Actor,self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.layers = nn.ModuleList()
        self.hidden_list = hidden_list
        self.lstm_hidden_size = 64
        self.lstm_layer = 2
        self.lstm = nn.LSTM(input_size =self.state_dim,
                            hidden_size = self.lstm_hidden_size, 
                            num_layers = self.lstm_layer,
                            batch_first = True,
                            bidirectional=True)  
        insize = self.lstm_hidden_size * 2    + 2  
        for outsize in self.hidden_list:
            fc = nn.Linear(insize,outsize)
            insize = outsize
            self.layers.append(fc)
        self.mean_layer = nn.Linear(insize,self.action_dim)
        self.log_std_layer = nn.Linear(insize, self.action_dim)
        
    def forward(self, state, moneyAndRatio):     
        money_ratio = moneyAndRatio[:,-1]
        stock_ratio = 1 - money_ratio
        h0 = torch.zeros(self.lstm_layer*2, state.size(0), self.lstm_hidden_size).to(device) 
        c0 = torch.zeros(self.lstm_layer*2, state.size(0), self.lstm_hidden_size).to(device)
        lstm_out, (hn, cn) = self.lstm(state,(h0,c0)) ## batch_first
        out = lstm_out[:,-1,:].view(lstm_out.size()[0],-1) 
        out = torch.cat((out, moneyAndRatio), -1)      
        for layer in self.layers:
            out = F.leaky_relu(layer(out),0.2,True)
        mean = self.mean_layer(out)   
        log_std = self.log_std_layer(out)
        log_std = torch.clamp(log_std, -20, 2)
        std = log_std.exp()
        normal_dist = Normal(mean, std)
        z = normal_dist.sample()
        true_action =  torch.tanh(z)
        
        zeros_action = torch.zeros_like(true_action).to(device)
        invest_action = torch.max(true_action, zeros_action) * money_ratio + torch.min(true_action, zeros_action) * stock_ratio
             
        return true_action, mean, log_std, invest_action
        
    



class LSTM_Q_net(nn.Module):
    def __init__(self, state_dim, time_steps=10,
                 action_dim=1,
                 hidden_list=[128,64]):
        super(LSTM_Q_net,self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.layers = nn.ModuleList()
        self.hidden_list = hidden_list
        self.lstm_hidden_size = 64
        self.lstm_layer =  2
        self.lstm = nn.LSTM(input_size =self.state_dim,
                            hidden_size = self.lstm_hidden_size, 
                            num_layers = self.lstm_layer,
                            batch_first = True, 
                            bidirectional=True)  
        insize = self.lstm_hidden_size * 2 + 3 
        for outsize in self.hidden_list:
            fc = nn.Linear(insize,outsize)
            insize = outsize
            self.layers.append(fc)
        self.q_out_layer = nn.Linear(insize,1)
        
    def forward(self, state, moneyAndRatio, actions):       
        h0 = torch.zeros(self.lstm_layer*2, state.size(0), self.lstm_hidden_size).to(device) 
        c0 = torch.zeros(self.lstm_layer*2, state.size(0), self.lstm_hidden_size).to(device)
        lstm_out, (hn,cn) = self.lstm(state,(h0,c0)) ## batch_first
        out = lstm_out[:,-1,:].view(lstm_out.size()[0],-1) 
        out = torch.cat((out, moneyAndRatio, actions),-1)
        for layer in self.layers:
            out = F.leaky_relu(layer(out),0.2,True)
        q_value = self.q_out_layer(out)       
        return q_value
