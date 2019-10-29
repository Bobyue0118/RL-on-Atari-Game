import gym
import keras
from keras.models import Sequential
from keras.layers import Dense,Flatten,Conv2D,BatchNormalization
import numpy as np
import random
from collections import deque 
from keras.optimizers import RMSprop ,Adam
import time 
import os 
import argparse
import sys
import os
from path import Path
import csv
import copy
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser(description='DQN for pixel game',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--pretrained-model', dest='pretrained_model', default=None, metavar='PATH',
                    help='path to pre-trained net model')
parser.add_argument('--name', dest='name', type=str, default='demo', required=True,
                    help='name of the experiment, checpoints are stored in checpoints/name')


class DQN:
    def __init__(self,env):
        self.env = env
        self.memory = deque(maxlen=500000)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min =0.005
        self.epsilon_decay = (self.epsilon-self.epsilon_min)/1000000  # 50000太小 反向傳播的梯度壓到1~-1 grident greeding 
                                                                    # model 問題

        self.batch_size = 32
        self.train_start =20000
        self.state_size = self.env.observation_space.shape[0]
        self.action_size =self.env.action_space.n
        self.learning_rate =0.00025

        self.evaluation_model = self.create_model()
        self.target_model = self.create_model()
       # print("state_size  ",self.state_size,"action_size  ",self.action_size)

    def create_model(self):
        model = Sequential()
        #model.add(Flatten())
        
        model.add(Dense(128,activation='relu', kernel_initializer="he_uniform"))
        model.add(Dense(128, activation='relu', kernel_initializer="he_uniform"))
        model.add(Dense(128, activation='relu', kernel_initializer="he_uniform"))
        model.add(Dense(128, activation='relu', kernel_initializer="he_uniform"))
        model.add(Dense(128, activation='relu', kernel_initializer="he_uniform"))

	#model.add(Dense(128, activation='relu', kernel_initializer="he_uniform"))
        #model.add(BatchNormalization())
        model.add(Dense(self.env.action_space.n,activation='linear', kernel_initializer="he_uniform"))
        optimizer = Adam(lr=self.learning_rate)
        model.compile(loss="mean_squared_error",optimizer=optimizer)
        return model 
    def choose_action(self,state,step_count):
        if self.epsilon > self.epsilon_min :
            if(step_count>0):
                self.epsilon -=self.epsilon_decay
            else:
                self.epsilon = self.epsilon_min
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        return np.argmax(self.evaluation_model.predict(state)[0])

    def target_train (self):    #fixed -Q targets
        self.target_model.set_weights(self.evaluation_model.get_weights())
        return
    def remember(self,cur_state, action, reward, new_state,done):
        self.memory.append((cur_state, action, reward, new_state,done))
        #print("Q : ",Q)
        return

    def replay(self):           #experience reply
        if len(self.memory) < self.train_start:
            return

        mini_batch = random.sample(self.memory, self.batch_size)

        update_input = np.zeros((self.batch_size,self.state_size))
        update_target = np.zeros((self.batch_size,self.action_size))

        for i in range(self.batch_size):
            state, action, reward, next_state, done = mini_batch[i]            
            target = self.evaluation_model.predict(state)[0]
            if done:
                target[action] = reward
            else:
                target[action] = reward + self.gamma*np.amax(self.target_model.predict(next_state)[0])
            
            update_input[i] = state
            update_target[i] = target
        self.evaluation_model.fit(update_input, update_target, batch_size = self.batch_size , epochs = 1 , verbose = 0)
        
def preprocess(x):
    return np.asarray([(n>>i)&1 for n in x for i in range(8)], dtype=np.bool)


def main():
    global args
    args = parser.parse_args() 
    save_path = Path(args.name)
    save_path = 'checkpoints'/save_path #/timestamp
    save_path.makedirs_p()
    training_writer = SummaryWriter(save_path)

    env = gym.make("Breakout-ram-v0")
    # print(type(env))
    episodes = 5000
    trial_len = 8000
    
    time_cost = time.time()
    batch_time_cost = time.time()
    batch_reward = 0
    sum_rewards = 0
    update_count = 0;
    step_count = 0;
    result=[]
    past_ = {"ale.lives":5}
    print("enter training")
    dqn_agent = DQN(env=env)
    for i_episode in range(episodes):
        total_reward = 0
        cur_state = env.reset().reshape(1,128)
        #cur_state = preprocess(cur_state).reshape(1,1024)
        #print(np.shape(cur_state))
        #print(cur_state)
        env.step(1)
        for step in range(trial_len):
            #os.system("pause")
            update_count+=1;
            step_count+=1;
            env.render('rgb_array')
            action = dqn_agent.choose_action(cur_state,step_count)
            new_state, reward, done, _ =env.step(action)
            #print(action)
            #print(reward)
            new_state = new_state.reshape(1,128)
            #new_state = preprocess(new_state).reshape(1,1024)

            if(past_['ale.lives']>_['ale.lives']):
                #print("past_['ale.lives']",past_['ale.lives'],"_['ale.lives']",_['ale.lives'])
                new_state1, reward1, done1, _1 =env.step(1)
                #new_state = new_state.reshape(1,128)
            
                total_reward += reward1
                batch_reward += reward1
                sum_rewards += reward1
            past_=_
            total_reward += reward
            batch_reward += reward
            sum_rewards += reward
            dqn_agent.remember(cur_state, action, reward, new_state,done)            

            if(step%4==0):
		#print("replay")
                dqn_agent.replay()

            cur_state = new_state
            
            if done :
                print("i_episode: [",i_episode,"]  ,  total_reward: [",total_reward,"]   ,   step : [",step,"]")             
                env.reset()
                if(update_count>=2500):
                    #print("target_train")
                    update_count-=2500
                    dqn_agent.target_train()
                if i_episode%100==99:
                    t = (time.time()-batch_time_cost)
                    print("--------average of ",i_episode," average reward : [",batch_reward/100,"]  ,time_use : [",t//60," min ",t-(t//60)*60,"'s]--------")
                    result.append([i_episode,batch_reward/100,t])
                    batch_time_cost = time.time();
                    batch_reward = 0              
                break
            if step==trial_len-1:
                print("*i_episode: [",i_episode,"]  ,  total_reward: [",total_reward,"]   ,   step : [",step,"]") 
                env.reset()
                if(update_count>=2500):
                    update_count-=2500
                    dqn_agent.target_train()
                if i_episode%100==99:
                    t = (time.time()-batch_time_cost)
                    print("*--------average of ",i_episode," average reward : [",batch_reward/100,"]  ,time_use : [",t//60," min ",t-(t//60)*60,"'s]--------")
                    batch_time_cost = time.time();
                    batch_reward = 0
    
        training_writer.add_scalar('Reward', total_reward, i_episode)
    # print("end of training, time_use:[ ",(time.time()-time_cost),"] , average sum_rewards : [",sum_rewards/5000,"]")
    for i,score, t in result :
       print("*--------average of ",i," average reward : [",score,"]  ,time_use : [",t//60," min ",t-(t//60)*60,"'s]--------")
    env.env.close()


if __name__ == '__main__':
    main()

