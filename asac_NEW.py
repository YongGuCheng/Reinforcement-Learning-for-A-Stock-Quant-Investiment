from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import os
import numpy as np
import pandas as pd
import tushare as ts
import backtrader as bt
from ASAC_Trader import ASAC_Trader_wrapper
import matplotlib.pyplot as plt
import copy

###############################  training  ####################################


KEEP_DAY = 6
TIME_STEPS = 55
STATE_DIM = 5
trader = ASAC_Trader_wrapper(state_dim=STATE_DIM, action_dim=1, time_steps=TIME_STEPS,eval_mode=True)

 
# Create a Stratey
class TestStrategy(bt.Strategy):
    params = dict(maperiod=55)
 
    def log(self, txt, dt=None):
        ''' Logging function for this strategy'''
        dt = dt or self.datas[0].datetime.date(0)
        print('%s, %s' % (dt.isoformat(), txt))
 
    def __init__(self):
        
            
        volume_ema = bt.indicators.EMA(self.data.volume, period=self.p.maperiod, plot=False)
        self.close_ema = bt.indicators.EMA(self.data.close, period=self.p.maperiod, plot=False)
        
        self.volume_ratio = self.data.volume / (volume_ema)
        self.close_ratio = self.data.close / (self.close_ema) 
        self.open_ratio = self.data.open / (self.close_ema) 
        self.high_ratio = self.data.high / (self.close_ema) 
        self.low_ratio = self.data.low / (self.close_ema) 
        
        

        self.state_list = []
        self.last_state_value = 0
        self.last_state = np.array([[]])
        self.time_steps = TIME_STEPS
        self.order = None
        self.data_store = []
        self.current_cold_day = 0
        
        self.last_stock_size = 0
        self.last_money = 0
        self.current_stock_size = 0
        self.avg_stock_cost = self.data.close[0]

    def notify_order(self, order):
        
        if order.status in [order.Submitted, order.Accepted]:
            return
        if order.status in [order.Completed]:
            if order.isbuy(): 
                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
            else:  
                pass
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log(order.status)
        self.order = None

        
#    def notify_cashvalue(self, cash, value):
#        self.log('Cash %s Value %s' % (cash, value)) 
    
#    def notify_trade(self, trade):
#        if not trade.isclosed:
#            return
#        self.log('OPERATION PROFIT, GROSS %.2f, NET %.2f' %
#                 (trade.pnl, trade.pnlcomm))
 
    def store_data(self, current_state, current_action, current_moneyRatio, chance_return):
        sg_data = dict(
                    current_state= current_state,
                    current_moneyRatio = current_moneyRatio,
                    current_action = current_action,
                    chance_return = chance_return
                    )
        self.data_store.append(sg_data)        

        if len(self.data_store) == 2:
            trader.total_step = trader.total_step + 1  # the trader add experience
            last_experience = self.data_store[0]
            next_experience = self.data_store[1]
            
            sg_return = 10 * next_experience['chance_return'] + 0.03 * np.abs(last_experience['current_action'])       ## a single reward for ASAC
            trader.add_experience( state=last_experience['current_state'],
                                 moneyRatio=last_experience['current_moneyRatio'], 
                                 action=last_experience['current_action'], 
                                 reward=sg_return, ## this is shifted
                                 next_state=next_experience['current_state'], 
                                 next_moneyRatio=next_experience['current_moneyRatio'], 
                                 done=[False])                    

            # delete first-line data after a transition-tuple is generated
            self.data_store = self.data_store[1:]


    def collect_status(self):            
        self.last_state_value = self.broker.getvalue()
        self.last_money = self.broker.getcash()
        self.last_stock_size = self.position.size
        if self.position.size > 0:
             self.avg_stock_cost = (self.broker.getvalue() - self.broker.getcash() ) / self.position.size

             
    def next(self):
        
        self.current_cold_day = self.current_cold_day - 1
        if self.order:
            return 
        
        average_cost_ratio = self.avg_stock_cost / self.close_ema[0]
        current_money = self.broker.getcash()
        current_value = self.broker.getvalue()
        money_ratio = current_money * 1.0 / (current_value)
        current_moneyRatio = np.array([ average_cost_ratio, money_ratio ])
        current_status =  [self.close_ratio[0],self.open_ratio[0],self.volume_ratio[0], \
                            self.high_ratio[0],self.low_ratio[0]]      
                                    
        self.state_list.append(current_status)
        self.state_list = self.state_list[-self.time_steps:]     
        current_state = np.array(self.state_list).reshape(1,-1,STATE_DIM)
        source_action, invest_action = trader.choose_action(current_state, current_moneyRatio)
        source_action, invest_action = source_action.item(), invest_action.item()

        if self.last_state.shape[1] < self.time_steps:
            self.last_state_value = self.broker.getvalue()
            self.last_state = copy.deepcopy(current_state)
            self.last_money = self.broker.get_cash()
            return   # making no action if data is not enough         
        else:
            if self.current_cold_day > 0:
                return   #  no action is made when cold
            else:
                self.current_cold_day = KEEP_DAY

                last_return = current_value * 1.0 / (self.last_state_value) - 1 
                if_no_action_value = self.last_stock_size * self.data.close[0] + self.last_money
                no_action_return = if_no_action_value * 1.0 / (self.last_state_value) - 1 
                chance_return = last_return - no_action_return
                
                # the following status processing should be before the action is excuted
                self.store_data(current_state, source_action, current_moneyRatio, chance_return)
                self.collect_status() 
                trader.trader_learn()
         
                # finally the actions for current_state is excuted
                if invest_action > 0:              
                    buy_size = int(invest_action * current_value / (self.data.close[0]) )
                    buy_size = int(buy_size * 0.90/100) * 100 ## in case of next open price > close_price          
                    self.order = self.buy(size=buy_size)
                    
                elif invest_action < 0:        
                    sell_size = int( - invest_action * (current_value ) / (self.data.close[0]) )
                    sell_size = int(sell_size/100 ) * 100
                    self.order = self.sell(size=sell_size) 

            

        



final_value_list = []
def run_train():
    
    data_folder = 'data'
    file_name_list = os.listdir(data_folder)
    np.random.shuffle(file_name_list)
    
    for file_index in range(len(file_name_list)):
            
        try:
            file_name = file_name_list[file_index]
            file_path = os.path.join(data_folder,file_name)
            dataframe = pd.read_csv(file_path) 
            dataframe.index = dataframe['date'].apply(lambda x: pd.Timestamp(x))
            dataframe = dataframe.where(dataframe != 0)
            dataframe = dataframe.fillna(method='ffill')
            dataframe['openinterest'] = 0 
            
            if dataframe.shape[0] < 1200:
                continue 
                        
            cerebro = bt.Cerebro()
            cerebro.addstrategy(TestStrategy)
            data = bt.feeds.PandasData(dataname=dataframe,                               
                                    fromdate=dataframe.index[-300].to_pydatetime(),                               
                                    todate=dataframe.index[-1].to_pydatetime()                              
                                    )

            cerebro.adddata(data)
            cerebro.broker.setcash(800000)
            cerebro.broker.set_shortcash(shortcash=False) ## this is important
            cerebro.broker.setcommission(commission=0.001)

            print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())
            cerebro.run()
            print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())
            reward = cerebro.broker.getvalue() * 1.0 / 800000 -1
            final_value_list.append(reward)
        except:
            print(file_name_list[file_index])
            continue
        cerebro.plot()

run_train()
    
plt.plot(final_value_list)
plt.show()
print(np.mean(final_value_list))
