#coding=utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import gym
from itertools import count
import matplotlib
import torch.optim as optim
import matplotlib.pyplot as plt


# Hyper Parameters
MEMORY_CAPACITY = 2000
BATCH_SIZE = 32
LR = 0.01                   # learning rate
EPSILON = 0.9               # greedy policy
TARGET_REPLACE_ITER = 100   # target update frequency



env = gym.make('CartPole-v0')
env = env.unwrapped

N_ACTIONS = env.action_space.n
N_STATES = env.observation_space.shape[0]


class Q_Net(nn.Module):
    def __init__(self):
        super(Q_Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 10)
        self.fc2 = nn.Linear(10, N_ACTIONS)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        actions_value = self.fc2(x)
        return actions_value


class Experience(object):

    def __init__(self):
        self.counter = 0                                         
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))
        

    def push(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        index = self.counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.counter += 1

    def sample(self):
        
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :N_STATES])
        b_a = torch.LongTensor(b_memory[:, N_STATES:N_STATES+1].astype(int))
        b_r = torch.FloatTensor(b_memory[:, N_STATES+1:N_STATES+2])
        b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:])

        return b_s,b_a,b_r,b_s_


def choose_action(x,Q):
    x = torch.unsqueeze(torch.FloatTensor(x), 0)
    if np.random.uniform() < EPSILON:   # 选最优动作
        actions_value = Q.forward(x)
        action = torch.max(actions_value, 1)[1].data.numpy()[0]     # return the argmax
    else:   # 选随机动作
        action = np.random.randint(0, N_ACTIONS)
    return action

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

Q = Q_Net().to(device)
Q_hat = Q_Net().to(device)
Q_hat.load_state_dict(Q.state_dict())
Q_hat.eval()
experience = Experience()
optimizer = optim.RMSprop(Q.parameters())
loss_func = nn.MSELoss()

print('Collecting experience...')
for i_episode in range(4000):
    s = env.reset()
    ep_r = 0
    while True:

        env.render()
        a = choose_action(s,Q)

        # take action
        s_, r, done, info = env.step(a)

        # modify the reward
        x, _, theta, _ = s_
        r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
        r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
        r = r1 + r2
    

        experience.push(s, a, r, s_)

        ep_r += r

        if experience.counter > MEMORY_CAPACITY:
            
            b_s,b_a,b_r,b_s_ = experience.sample()
            
            q_eval = Q(b_s).gather(1, b_a)  # shape (batch, 1)
            q_next = Q_hat(b_s_).detach()     # detach from graph, don't backpropagate
            q_target = b_r + q_next.max(1)[0].unsqueeze(1)   # shape (batch, 1)

            loss = loss_func(q_eval, q_target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        if done:
            print('Episode: ', i_episode,'| Total Reward: ', round(ep_r, 2))
            break

        s = s_

    if i_episode % TARGET_REPLACE_ITER == 0:
        Q_hat.load_state_dict(Q.state_dict())



print('Complete')
env.render()
env.close()
        