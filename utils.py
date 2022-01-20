import gym
import numpy as np
import torch

def evaluate_policy(policy, env_name, seed, eval_episodes=3):
    eval_env = gym.make(env_name)
    eval_env.seed(seed + 100)
    avg_reward = 0.
    for _ in range(eval_episodes):
        state = False 
        done = eval_env.reset()
        while not done:
            action = policy.select_action(np.array(state))
            state, reward, done, _ = eval_env.step(action)
            avg_reward += reward
    avg_reward /= eval_episodes
    return avg_reward

class ReplayBuffer(object):
    
    def __init__(self, state_dimension, action_dimension, max_size=int(1e6)):
        self.capacity = max_size
        self.tailptr = 0
        self.size = 0
        
        self.state = np.zeros((max_size, state_dimension))
        self.action = np.zeros((max_size, action_dimension))
        self.next_state = np.zeros((max_size, state_dimension))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def update(self, state, action, next_state, reward, done):
        self.state[self.tailptr] = state
        self.action[self.tailptr] = action
        self.next_state[self.tailptr] = next_state
        self.reward[self.tailptr] = reward
        self.not_done[self.tailptr] = 1. - done
        
        self.tailptr = (self.tailptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        indice = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.FloatTensor(self.state[indice]).to(self.device),
            torch.FloatTensor(self.action[indice]).to(self.device),
            torch.FloatTensor(self.next_state[indice]).to(self.device),
            torch.FloatTensor(self.reward[indice]).to(self.device),
            torch.FloatTensor(self.not_done[indice]).to(self.device)
        )
        
# Ornstein-Ulhenbeck Process
# https://github.com/vitchyr/rlkit/blob/master/rlkit/exploration_strategies/ou_strategy.py
class OUNoise(object):
    def __init__(self, action_space, mu=0.0, theta=0.15, max_sigma=0.2, min_sigma=0.2, decay_period=100000):
        self.mu           = mu
        self.theta        = theta
        self.sigma        = max_sigma
        self.max_sigma    = max_sigma
        self.min_sigma    = min_sigma
        self.decay_period = decay_period
        self.action_dim   = action_space.shape[0]
        self.low          = action_space.low
        self.high         = action_space.high
        self.reset()
        
    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu
        
    def evolve_state(self):
        x  = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state
    
    def get_action(self, action, t=0): 
        ou_state = self.evolve_state()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
        return np.clip(action + ou_state, self.low, self.high)
    

def evaluate_policy(policy, env_name, seed, eval_episodes=3, render=False):
    eval_env = gym.make(env_name)
    eval_env.seed(seed + 100)
    avg_reward = 0.
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        while not done:
            action = policy.select_action(np.array(state))
            if render:
                eval_env.render()
            state, reward, done, _ = eval_env.step(action)
            avg_reward += reward
    avg_reward /= eval_episodes
    return avg_reward

def normal_noise_action(action, max_action, action_dim):
    # print(action, max_action)
    action = (action + np.random.normal(0, max_action * 0.1, size=action_dim)).clip(-max_action, max_action)
    return action