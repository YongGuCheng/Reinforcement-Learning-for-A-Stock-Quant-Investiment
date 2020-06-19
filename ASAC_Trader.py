import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from utils.util import LSTM_ReplayBuffer
from utils.LSTM_models import LSTM_SAC_Actor, LSTM_Q_net
from torch.distributions import Normal

torch.manual_seed(666)
device = torch.device("cpu")     

class ASAC_Trader_wrapper(object):
    def __init__(self, state_dim, 
			action_dim, 
                        time_steps=10,
                        eval_mode=False,
                        gamma=0.99,
                        tau=0.05,
                        batch_size=64,
                        buffer_size=40000,
                        start_learn_step=10000,
                       ):
        self.trader_model_path = './weights/LSTM_ASAC/trader_model.pth'
        self.q_net1_model_path = './weights/LSTM_ASAC/q_net1_model.pth'
        self.q_net2_model_path = './weights/LSTM_ASAC/q_net2_model.pth'
        
        
        self.action_dim, self.state_dim, self.time_steps = action_dim, state_dim, time_steps
        self.batch_size, self.buffer_size, self.gamma, self.tau = batch_size, buffer_size, gamma, tau
        self.replay_buffer = LSTM_ReplayBuffer(self.state_dim, self.action_dim, self.buffer_size, time_steps=time_steps)
        self.total_step, self.start_learn_step =  0, start_learn_step
        self.eval_mode= eval_mode
        
        self.trader = LSTM_SAC_Actor(self.state_dim, self.action_dim).to(device)
        self.Q_net1 = LSTM_Q_net(self.state_dim, self.action_dim).to(device)
        self.Q_net2 = LSTM_Q_net(self.state_dim, self.action_dim).to(device)
        self.Q_net_target1 = LSTM_Q_net(self.state_dim, self.action_dim).to(device)
        self.Q_net_target2 = LSTM_Q_net(self.state_dim, self.action_dim).to(device)

       
        self.Q_net1_criterion = nn.MSELoss()
        self.Q_net2_criterion = nn.MSELoss()
    
        self.q_net_parameter = list(self.Q_net1.parameters()) + list(self.Q_net2.parameters())
    
        self.trader_optimizer = optim.Adam(self.trader.parameters(), lr=0.0003) 
        self.Qnet_optimizer = optim.Adam(self.q_net_parameter, lr=0.0003)        
        
        
        self.target_entropy = - np.prod((self.action_dim,)).item()
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device).to(device)
        self.alpha = 0.2
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=0.0003)
        
        self.load_model()
        
        
        
        
    def choose_action(self, state, moneyAndRatio):
        state = torch.FloatTensor(state).to(device)
        moneyAndRatio = torch.FloatTensor(moneyAndRatio).unsqueeze(0).to(device)
        true_action, _,  _, invest_action = self.trader(state, moneyAndRatio)
        return true_action.cpu().detach().numpy(), invest_action.cpu().detach().numpy() 


    def evaluate(self, state, moneyAndRatio ):
        _, mean,  log_std, _ = self.trader(state, moneyAndRatio)
        std = log_std.exp()
        normal_dist = Normal(mean, std)
        dist = Normal(0,1)
        z = dist.sample().to(device)
        action_tanh = torch.tanh( mean + std * z )
        log_prob = normal_dist.log_prob(mean + std * z) - torch.log( 1 -  action_tanh**2  + 1e-8 )
        return action_tanh, log_prob
        

    def add_experience(self, state, moneyRatio, action, reward, next_state, next_moneyRatio, done):
        self.replay_buffer.add(state, moneyRatio, action, reward, 
                               next_state, next_moneyRatio, done)

    def soft_update(self):
        for params, target_params in zip(self.Q_net1.parameters(),self.Q_net_target1.parameters()):
            target_params.data.copy_(params * self.tau + target_params * (1 - self.tau))

        for params, target_params in zip(self.Q_net2.parameters(),self.Q_net_target2.parameters()):
            target_params.data.copy_(params * self.tau + target_params * (1 - self.tau))
            
    def trader_learn(self):
        if self.total_step > self.start_learn_step and not self.eval_mode:
            batch = self.replay_buffer.sample(self.batch_size) 
            state1, state2, actions, rewards, _ = batch['obs1'],batch['obs2'],batch['acts'],batch['rews'],batch['done']
            moneyAndRatio1, next_moneyAndRatio = batch['moneyRatio'], batch['next_moneyRatio']

            q1 = self.Q_net1(state1, moneyAndRatio1, actions)
            q2 = self.Q_net2(state1, moneyAndRatio1, actions)

            next_pi, next_pi_log_prob = self.evaluate(state2, next_moneyAndRatio)
            next_pi_q1 = self.Q_net_target1(state2, next_moneyAndRatio, next_pi)
            next_pi_q2 = self.Q_net_target2(state2, next_moneyAndRatio, next_pi)
            next_pi_q = torch.min(next_pi_q1, next_pi_q2)

            next_v_value = next_pi_q - self.alpha * next_pi_log_prob
            q_target = rewards + self.gamma * next_v_value
            
            
            q1_loss = self.Q_net1_criterion(q1, q_target.detach())
            q2_loss = self.Q_net2_criterion(q2, q_target.detach())
            q_net_loss = q1_loss + q2_loss
            
            
            pi,  pi_log_prob  = self.evaluate(state1, moneyAndRatio1)
            pi_q1 = self.Q_net_target1(state1, moneyAndRatio1, pi)
            pi_q2 = self.Q_net_target2(state1, moneyAndRatio1, pi)
            pi_q = torch.min(pi_q1, pi_q2)
            trader_loss = (self.alpha * pi_log_prob - pi_q).mean()


            alpha_loss = - (self.log_alpha * (pi_log_prob + self.target_entropy).detach()).mean()


            self.Qnet_optimizer.zero_grad()
            q_net_loss.backward()
            #nn.utils.clip_grad_norm_(self.Q_net1.parameters(), 0.5)
            self.Qnet_optimizer.step()
            
            
            self.trader_optimizer.zero_grad()
            trader_loss.backward()
            #nn.utils.clip_grad_norm_(self.trader.parameters(), 0.5)
            self.trader_optimizer.step()


            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            
            self.alpha = self.log_alpha.exp()

            
            self.soft_update()
            
            if (self.total_step + 1) % 10000 == 0:
                self.save_model()

    def save_model(self):
        torch.save(self.trader,self.trader_model_path)
        torch.save(self.Q_net_target1,self.q_net1_model_path)
        torch.save(self.Q_net_target2,self.q_net2_model_path)
   
    def load_model(self):
        try:
            self.trader.load_state_dict(torch.load(self.trader_model_path,map_location=device).state_dict())
            self.Q_net1.load_state_dict(torch.load(self.q_net1_model_path,map_location=device).state_dict())
            self.Q_net2.load_state_dict(torch.load(self.q_net2_model_path,map_location=device).state_dict())
            self.Q_net_target1.load_state_dict(torch.load(self.q_net1_model_path,map_location=device).state_dict())
            self.Q_net_target2.load_state_dict(torch.load(self.q_net2_model_path,map_location=device).state_dict())
            
            print('load model success')
        except:
            print('load model error')


