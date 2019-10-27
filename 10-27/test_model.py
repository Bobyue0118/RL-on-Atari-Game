import argparse
import sys
import os
from path import Path


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


class DQN:
    ### TUNE CODE HERE ###
    def __init__(self, env):
        """
        env = gym.make('Breakout-ram-v0')
        env = env.unwrapped
        """
        self.env = env
        self.memory = deque(maxlen=400000) # 双向队列
        self.gamma = 0.8
        self.epsilon = 0.2
        self.epsilon_min = 0.01
        self.epsilon_decay =  self.epsilon_min / 5000 # 
        
        self.batch_size = 32
        self.train_start = 1000
        self.state_size = self.env.observation_space.shape[0]*4
        self.action_size = self.env.action_space.n
        self.learning_rate = 0.005
        
        self.evaluation_model = self.create_model() 
        self.target_model = self.create_model()
        
    def create_model(self):
        model = Sequential() # The Sequential model is a linear stack of layers.
        model.add(Dense(128*2, input_dim=self.state_size,activation='relu'))
        model.add(Dense(128*2, activation='relu'))
        model.add(Dense(128*2, activation='relu'))
        model.add(Dense(self.env.action_space.n, activation='linear'))# self.env.action_space.n = 4
        model.compile(loss='mean_squared_error', optimizer=optimizers.RMSprop(lr=self.learning_rate,decay=0.99,epsilon=1e-6))
        return model
    
    def choose_action(self, state, steps):
        
        # 取出a中元素最大值所对应的索引
        return np.argmax(self.evaluation_model.predict(state)[0])


    def binary_encoding(self,decimal_state):
        binary_state = [np.binary_repr(x,width=8) for x in decimal_state[0]]
        parameter_list = []
        for parameter in binary_state:
            parameter_list.append(int(parameter[:2],2))
            parameter_list.append(int(parameter[2:4],2))
            parameter_list.append(int(parameter[4:6],2))
            parameter_list.append(int(parameter[6:],2))
        return np.array(parameter_list)


    
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

    env = gym.make('Breakout-ram-v0')
    env = env.unwrapped
    
    episodes = 1

    
    tmp_reward=0
    sum_rewards = 0
    
    graph_reward = []
    graph_episodes = []
    
    dqn_agent = DQN(env=env)

    ####### Training ######
    ### START CODE HERE ###

    dqn_agent.evaluation_model = load_model(args.pretrained_model)


    current_step = 0
    for episode in range(episodes):
        prev_state = env.reset().reshape(1,128)
        prev_state = dqn_agent.binary_encoding(prev_state).reshape(1,512)
        # print(prev_state)
        
        reward_in_episode = 0
        print('episode:',episode)
        while True:
            env.render()
            action = dqn_agent.choose_action(prev_state,current_step)
            current_state, reward, done, _ = env.step(action)  #  [1,128]
            current_state = current_state.reshape(1, 128)
            current_state = dqn_agent.binary_encoding(current_state).reshape(1,512)
            
            current_step += 1
            reward_in_episode += reward

            prev_state = current_state
            
            if done:
                env.render()
                break
        # visualization
        graph_reward.append(reward_in_episode)
        graph_episodes.append(episode)


    ### END CODE HERE ###
    
    dqn_agent.visualize(graph_reward, graph_episodes)
    
if __name__ == '__main__':
    main()