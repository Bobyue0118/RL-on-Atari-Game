import argparse
import sys
import os
from path import Path
import csv
import copy


import random
import numpy as np
import pandas as pd
import tensorflow as tf
import gym
from collections import deque
from keras import optimizers
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Activation, Flatten, Conv1D, MaxPooling1D,Reshape
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
from tqdm import tqdm

parser = argparse.ArgumentParser(description='DQN for pixel game',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--pretrained-model', dest='pretrained_model', default=None, metavar='PATH',
                    help='path to pre-trained net model')
parser.add_argument('--name', dest='name', type=str, default='demo', required=True,
                    help='name of the experiment, checpoints are stored in checpoints/name')

class DQN:
    ### TUNE CODE HERE ###
    def __init__(self, env):
        """
        env = gym.make('Breakout-ram-v0')
        env = env.unwrapped
        """
        self.env = env
        self.memory = deque(maxlen=400000) # 双向队列
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.005
        self.epsilon_decay =  (self.epsilon-self.epsilon_min)/1000000 # 
        
        self.batch_size = 32
        self.train_start = 20000
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        self.learning_rate = 0.00025
        
        self.evaluation_model = self.create_model() 
        self.target_model = self.create_model()
        
    def create_model(self):
        model = Sequential() # The Sequential model is a linear stack of layers.
        model.add(Dense(128, input_dim=self.state_size,activation='relu'))
        model.add(Dense(128*4, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.env.action_space.n, activation='linear'))# self.env.action_space.n = 4
        model.compile(loss='mean_squared_error', optimizer=optimizers.Adam(lr=self.learning_rate))
        return model
    
    def choose_action(self, state, steps):
        if steps > 10000:
            if self.epsilon > self.epsilon_min:
                self.epsilon -= self.epsilon_decay
        if np.random.random() < self.epsilon: # 小于epsilon，随机采取措施进行探索
            return self.env.action_space.sample()
            # self.evaluation_model.predict(state) 就是网络的输出，4个元素
        
        # 取出a中元素最大值所对应的索引
        return np.argmax(self.evaluation_model.predict(state)[0])
        
    def remember(self, cur_state, action, reward, new_state, done):
        '''
        用于记录所有的决策以及结果
        '''
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        
        transition = (cur_state, action, reward, new_state, done)
        self.memory.extend([transition]) # add to right，双向队列，加到右边
        # ([(1, 2, 3), (2, 3, 4), (10, 11, 12)], maxlen=400000) 大概是这样
        self.memory_counter += 1
    
    def replay(self):
        '''
            相当于模拟训练过程，加速收敛
        '''
        if len(self.memory) < self.train_start:
            #队列中的元素少于1000时直接返回
            return
        
        mini_batch = random.sample(self.memory, self.batch_size) # 试了下，就是随机选取的 (cur_state, action, reward, new_state, done)
        
        update_input = np.zeros((self.batch_size, self.state_size))
        update_target = np.zeros((self.batch_size, self.action_size))
        
        for i in range(self.batch_size):
            state, action, reward, new_state, done = mini_batch[i]
            target = self.evaluation_model.predict(state)[0]
        
            if done:
                target[action] = reward
            else:
                target[action] = reward + self.gamma * np.amax(self.target_model.predict(new_state)[0])
            
            update_input[i] = state
            update_target[i] = target
    
        self.evaluation_model.fit(update_input, update_target, batch_size=self.batch_size, epochs=1, verbose=0)
    
    def target_train(self):
        self.target_model.set_weights(self.evaluation_model.get_weights())
        return
    
    def binary_encoding(self,decimal_state):
        binary_state = [np.binary_repr(x,width=8) for x in decimal_state[0]]
        parameter_list = []
        for parameter in binary_state:
            parameter_list.append(int(parameter[:2],2))
            parameter_list.append(int(parameter[2:4],2))
            parameter_list.append(int(parameter[4:6],2))
            parameter_list.append(int(parameter[6:],2))
        return np.array(parameter_list)

    def train_with_gt(self):       
        csv_file=open('labeled_dataset.csv')
        csv_reader_lines = csv.reader(csv_file)
        for line in csv_reader_lines:
            line = list(map(int, line))
            update_input = np.array(line[:-4]).reshape(1,128)
            update_target = np.array(line[-4:]).reshape(1,4)
            self.evaluation_model.fit(update_input, update_target, batch_size=1, epochs=1, verbose=0)
            


    
    def visualize(self, reward, episode):
        plt.plot(episode, reward, 'ob-')
        plt.title('Average reward each 100 episode')
        plt.ylabel('Reward')
        plt.xlabel('Episodes')
        plt.grid()
        plt.show()
    ### END CODE HERE ###
    
        
def main():
    global args
    args = parser.parse_args() 
    save_path = Path(args.name)
    save_path = 'checkpoints'/save_path #/timestamp
    save_path.makedirs_p()
    training_writer = SummaryWriter(save_path)


    env = gym.make('Breakout-ram-v0')
    env = env.unwrapped
    
    episodes = 5000
    trial_len = 10000
    
    
    graph_reward = []
    graph_episodes = []
    
    dqn_agent = DQN(env=env)
    for _ in range(3):
        dqn_agent.train_with_gt()

    ####### Training ######
    ### START CODE HERE ###
    dqn_agent.create_model()
    current_step = 0
    for episode in range(episodes):
        prev_state = env.reset().reshape(1,128)
        # prev_state = dqn_agent.binary_encoding(prev_state).reshape(1,512)
        # print(prev_state)
        
        reward_in_episode = 0
        update_count = 0
        prev_lives = {"ale.lives":5}
        lives = {}
        current_state = 0
        reward_100episodes = 0
        env.step(1)
        print('episode:',episode)
        for _ in tqdm(range(trial_len)):
            # env.render()
            action = dqn_agent.choose_action(prev_state,current_step)
            
            del current_state, lives
            current_state, reward, done, lives = env.step(action)  #  [1,128]

            current_state = current_state.reshape(1, 128)
            # current_state = dqn_agent.binary_encoding(current_state).reshape(1,512)
            
            current_step += 1
            reward_in_episode += reward
            reward_100episodes += reward
            update_count += 1

            dqn_agent.remember(copy.deepcopy(prev_state), action, reward, copy.deepcopy(current_state), done)

            if (prev_lives['ale.lives']>lives['ale.lives']):
                del prev_lives
                prev_lives = copy.deepcopy(lives)
                action = np.int64(1)
                current_state, reward, done, lives = env.step(action)
                current_state = current_state.reshape(1, 128)

            if current_step % 4==0:
                dqn_agent.replay()

            del prev_state
            
            prev_state = copy.deepcopy(current_state)

            
            if done:
                if(update_count>=300) or current_step%2500==0:
                    dqn_agent.target_train()
                # env.render()
                


                if episode % 100 == 99:
                    reward_100episodes
                    reward_100episodes = 0
                break
        # visualization
        if episode%500 ==0:           
            dqn_agent.target_model.save(save_path + '/my_model.h5')

        training_writer.add_scalar('Reward', reward_in_episode, episode)
        training_writer.add_scalar('Reward for 100episodes', reward_100episodes, episode)
        graph_reward.append(reward_in_episode)
        graph_episodes.append(episode)


    ### END CODE HERE ###
    dqn_agent.target_model.save(save_path + '/my_model.h5')
    dqn_agent.visualize(graph_reward, graph_episodes)
    
if __name__ == '__main__':
    main()