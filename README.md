# Reproduction-of-Continuous-Control-with-Deep-Reinforcement-Learning-

This project is the reproduction of  [DDPG](https://arxiv.org/pdf/1509.02971v6.pdf).

## Run DDPG

```shell
conda env create -f environment.yml
conda activate myddpg

mkdir exp
mkdir exp/log
mkdir exp/summary
mkdir exp/checkpoint
mkdir exp/visualization

chmod +x ./scripts/tn
./scripts/tn.sh
# for visualization in tensorboard
tensorboard --logdir='exp/summary' --port=6006
```

## Visualization

### Rewards with training step

<img src="Reproduction of Continuous Control with Deep Reinforcement Learning.assets/rewards.png" alt="rewards" style="zoom:80%;" />

<img src="Reproduction of Continuous Control with Deep Reinforcement Learning.assets/rewards-16426773689861.png" alt="rewards" style="zoom:80%;" />

<img src="Reproduction of Continuous Control with Deep Reinforcement Learning.assets/rewards-16426773748242.png" alt="rewards" style="zoom:80%;" />

### Density plot showing estimated Q values versus observed returns sampled from test episodes

<img src="Reproduction of Continuous Control with Deep Reinforcement Learning.assets/density.png" alt="density" style="zoom:80%;" />

<img src="Reproduction of Continuous Control with Deep Reinforcement Learning.assets/density-16426774664443.png" alt="density" style="zoom:80%;" />

<img src="Reproduction of Continuous Control with Deep Reinforcement Learning.assets/density-16426774715944.png" alt="density" style="zoom:80%;" />

## References

https://github.com/vitchyr/rlkit/blob/master/rlkit/exploration_strategies/ou_strategy.py

https://arxiv.org/pdf/1509.02971v6.pdf

http://proceedings.mlr.press/v32/silver14.pdf

https://www.datascienceassn.org/sites/default/files/Human-level%20Control%20Through%20Deep%20Reinforcement%20Learning.pdf
