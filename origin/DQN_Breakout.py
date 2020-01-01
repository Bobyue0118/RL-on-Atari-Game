import random
import numpy as np
import pandas as pd
import tensorflow as tf
import gym
from collections import deque
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Activation, Flatten, Conv1D, MaxPooling1D,Reshape
import matplotlib.pyplot as plt

class DQN:
    ### TUNE CODE HERE ###
    def __init__(self, env):
        self.env = env
        self.memory = deque(maxlen=400000)
        self.gamma = 0.8
        self.epsilon = 0.2
        self.epsilon_min = 0.01
        self.epsilon_decay =  self.epsilon_min / 5000
        
        self.batch_size = 32
        self.train_start = 1000
        self.state_size = self.env.observation_space.shape[0]*4
        self.action_size = self.env.action_space.n
        self.learning_rate = 0.005
        
        self.evaluation_model = self.create_model()
        self.target_model = self.create_model()
        
    def create_model(self):
        model = Sequential()
        model.add(Dense(128*2, input_dim=self.state_size,activation='relu'))
        model.add(Dense(128*2, activation='relu'))
        model.add(Dense(128*2, activation='relu'))
        model.add(Dense(self.env.action_space.n, activation='linear'))
        model.compile(loss='mean_squared_error', optimizer=optimizers.RMSprop(lr=self.learning_rate,decay=0.99,epsilon=1e-6))
        return model
    
    def choose_action(self, state, steps):
        if steps > 50000:
            if self.epsilon > self.epsilon_min:
                self.epsilon -= self.epsilon_decay
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        return np.argmax(self.evaluation_model.predict(state)[0])
        
    def remember(self, cur_state, action, reward, new_state, done):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        
        transition = (cur_state, action, reward, new_state, done)
        self.memory.extend([transition])
        
        self.memory_counter += 1
    
    def replay(self):
        if len(self.memory) < self.train_start:
            return
        
        mini_batch = random.sample(self.memory, self.batch_size)
        
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
    
    def visualize(self, reward, episode):
        plt.plot(episode, reward, 'ob-')
        plt.title('Average reward each 100 episode')
        plt.ylabel('Reward')
        plt.xlabel('Episodes')
        plt.grid()
        plt.show()
    ### END CODE HERE ###
    
        
def main():
    env = gym.make('Breakout-ram-v0')
    env = env.unwrapped
    
    episodes = 5000
    trial_len = 10000
    
    tmp_reward=0
    sum_rewards = 0
    
    graph_reward = []
    graph_episodes = []
    
    dqn_agent = DQN(env=env)

    ####### Training ######
    ### START CODE HERE ###






    ### END CODE HERE ###

    dqn_agent.visualize(graph_reward, graph_episodes)
    
if __name__ == '__main__':
    main()