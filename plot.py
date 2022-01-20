import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import numpy as np

# variant
variant_list = ['bn', 'tn', 'bntn', 'normalnoise']
colors = ['lightgrey', 'gray', 'green', 'blue']
envName='Hopper-v2'
# LunarLanderContinuous-v2
# Hopper-v2
# Pendulum-v1
checkpointPath = 'exp/checkpoint'

for variant, color in zip(variant_list, colors):
    variantPath = os.path.join(checkpointPath, "0119_" + variant + '_' + envName + "_eps1000000")
    with open(os.path.join(variantPath, 'evaluation_rewards'), "rb") as fp:   # Unpickling
        evaluation_rewards = pickle.load(fp)
    with open(os.path.join(variantPath, 'evaluation_steps'), "rb") as fp:   # Unpickling
        evaluation_steps = pickle.load(fp)
    evaluation_rewards = np.array(evaluation_rewards)
    rewards_avg = []
    for i in range(len(evaluation_rewards)):
        rewards_avg.append(np.mean(evaluation_rewards[max(0, i-10):i+1]))
    plt.plot(evaluation_steps, rewards_avg, c=color)
plt.xlabel('Step Million')
plt.ylabel('Reward')
plt.title(envName)
plt.legend(labels=['bn', 'tn', 'bn+tn', 'normalnoise'])
outputPath = os.path.join('exp/visualization/variants', envName)
if not os.path.exists(outputPath):
    os.mkdir(outputPath)
plt.savefig(fname='{}/{}'.format(outputPath, 'rewards'+'.png'))
# plt.show()
plt.close()

# density
cmap = sns.color_palette("Greys", as_cmap=True)
estimated_rewards = []
sns.set_style('whitegrid') 
variantPath = os.path.join(checkpointPath, "0119_" + "tn" + '_' + envName + "_eps1000000")
with open(os.path.join(variantPath, 'evaluation_rewards'), "rb") as fp:   # Unpickling
    evaluation_rewards = pickle.load(fp)
with open(os.path.join(variantPath, 'estimated_rewards'), "rb") as fp:   # Unpickling
    raw_estimated_rewards = pickle.load(fp)
    # print(raw_estimated_rewards)
    for arr in raw_estimated_rewards:
        estimated_rewards.append(arr.item())
        
splot = sns.kdeplot(x=evaluation_rewards, y=estimated_rewards, shade=True, cmap=cmap)
splot.plot([0,1], [0,1], ':y', c='black', transform=splot.transAxes) 
splot.set(ylabel='Estimated Q')
splot.set(xlabel='Return')
splot.set(title=envName)
sfig = splot.get_figure() 
outputPath = os.path.join('exp/visualization/density', envName)
if not os.path.exists(outputPath):
    os.mkdir(outputPath)
sfig.savefig(fname='{}/{}'.format(outputPath, 'density'+'.png'),  orientation="landscape")   
plt.close()