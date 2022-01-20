from matplotlib import pyplot as plt
import os
import sys
import logging
import gym
import pickle
import torch
import numpy as np
from tensorboardX import SummaryWriter
from tqdm import tqdm
from utils import *
from agents import *
from config import arg_parse, show_args, logger

np.set_printoptions(precision=4, suppress=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Argument Parse
args = arg_parse()
assert args.mode == 'bn'
# Bulid Logger
formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
file_path = 'exp/log/{}.log'.format(args.post)
file_handler = logging.FileHandler(file_path)
file_handler.setFormatter(formatter)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Show Argument
show_args(args)

writer = SummaryWriter('{}/{}'.format('exp/summary/', args.post))

env = gym.make(args.envName)
logger.info("action space shape:{}, max action:{}, min action:{}".format(
    env.action_space.shape, env.action_space.high[0], env.action_space.low[0]))

env.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)

state_dimension = env.observation_space.shape[0]
action_dimension = env.action_space.shape[0]
max_action = float(env.action_space.high[0])
agent = DDPGAgent(state_dim=state_dimension, action_dim=action_dimension,
                  max_action=max_action, device=device, discount=0.99, tau=5e-3,  mode=args.mode)
noise = OUNoise(env.action_space)

memory = ReplayBuffer(state_dimension, action_dimension)

# train
state = env.reset()
noise.reset()
done = False
episode_reward = 0
episode_timesteps = 0
episode_num = 0
evaluation_rewards = []
estimated_rewards = []
evaluation_steps = []
episode_rewards = []
episode_avg_rewards = []
max_evaluation_reward = -float('inf')
max_episode = 0

outputPath = os.path.join('exp/checkpoint/', args.post)
if not os.path.exists(outputPath):
    os.mkdir(outputPath)
    
for step in tqdm(range(1, int(args.maxSteps) + 1)):
    episode_timesteps += 1
    if step < args.warmSteps:
        action = env.action_space.sample()
    else:
        action = agent.select_action(state)
        action = noise.get_action(action, step)
        
    next_state, reward, done, _ = env.step(action)
    memory.update(state, action, next_state, reward,
        float(done) if episode_timesteps < env._max_episode_steps else 0)
    state = next_state
    episode_reward += reward
    
    if step >= args.warmSteps:
        estimated_reward = agent.train(memory, args.batchSize).data.cpu().numpy()
        
    if done:
        episode_rewards.append(episode_reward)
        episode_avg_rewards.append(np.mean(episode_rewards[-10:]))
        logger.info("Episode: {}, Episode_Reward: {}, Average_Episode_Reward: {}".format(episode_num, np.round(episode_reward, decimals=2), episode_avg_rewards[-1]))
        writer.add_scalar('Episode_Reward', episode_reward, episode_num)
        writer.add_scalar('Average_Episode_Reward', episode_avg_rewards[-1], episode_num)
        
        state = env.reset()
        noise.reset()
        done = False
        episode_reward = 0
        episode_timesteps = 0
        episode_num += 1
        
    if step % args.evaluateFreq == 0:
        evaluation_reward = evaluate_policy(agent, args.envName, args.seed) 
        evaluation_rewards.append(evaluation_reward)
        evaluation_steps.append(step)
        estimated_rewards.append(estimated_reward)
        writer.add_scalar('Evaluation_Reward', evaluation_reward, step)
        writer.add_scalar('Estimated_Reward', estimated_reward, step)
        writer.add_scalar('Average_Evaluation_Reward', np.mean(evaluation_rewards[-10:]), step)
        writer.add_scalar('Average_Estimated_Reward', np.mean(estimated_rewards[-10:]), step)
        agent.save_checkpoint(os.path.join(outputPath, args.envName))
        
        if evaluation_reward > max_evaluation_reward:
            max_evaluation_reward = evaluation_reward
            max_episode = episode_num
            agent.save_checkpoint(os.path.join(outputPath, 'best_' + args.envName))
            logger.info('[Best] [Episode {0}]: Best evaluation rewards is {1:.4f}'.format(max_episode, max_evaluation_reward))

with open(os.path.join(outputPath, "evaluation_rewards"), "wb") as fp:   #Pickling
   pickle.dump(evaluation_rewards, fp)
with open(os.path.join(outputPath, "evaluation_steps"), "wb") as fp:   #Pickling
   pickle.dump(evaluation_steps, fp)
with open(os.path.join(outputPath, "episode_rewards"), "wb") as fp:   #Pickling
   pickle.dump(evaluation_rewards, fp)
with open(os.path.join(outputPath, "estimated_rewards"), "wb") as fp:   #Pickling
   pickle.dump(estimated_rewards, fp)
logger.info('[Best] [Episode {0} of {2}]: Best evaluation rewards is {1:.4f}'.format(max_episode, max_evaluation_reward, episode_num))        

plt.plot(episode_rewards)
plt.plot(episode_avg_rewards)
plt.plot()
plt.xlabel('Episode')
plt.ylabel('Reward')
outputPath = os.path.join('exp/visualization/', args.post)
if not os.path.exists(outputPath):
    os.mkdir(outputPath)
plt.savefig(fname='{}/{}'.format(outputPath, 'train'+'.png'))
# plt.show()
plt.close()


plt.plot(evaluation_steps, evaluation_rewards)
plt.plot()
plt.xlabel('Step')
plt.ylabel('Reward')
outputPath = os.path.join('exp/visualization/', args.post)
if not os.path.exists(outputPath):
    os.mkdir(outputPath)
plt.savefig(fname='{}/{}'.format(outputPath, 'eval'+'.png'))
# plt.show()
plt.close()