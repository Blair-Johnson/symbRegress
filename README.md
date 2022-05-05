# symbRegress
Symbolic Regression term project for ECE 6254. This repository implements a deep reinforcement learning system for solving symbolic regression tasks.

# Usage
```
usage: train.py [-h] [--num_episodes NUM_EPISODES] [--max_depth MAX_DEPTH]
                [--min_depth MIN_DEPTH] [--logdir LOGDIR]

optional arguments:
  -h, --help            show this help message and exit
  --num_episodes NUM_EPISODES
  --max_depth MAX_DEPTH
  --min_depth MIN_DEPTH
  --logdir LOGDIR
```

# Requirements
```
numpy==1.21.5
torch==1.11.0
tensorboard==1.15.0
```

# Attribution
This project is loosely based on the work of Peterson et al.
```
@article{petersen2019deep,
  title={Deep symbolic regression: Recovering mathematical expressions from data via risk-seeking policy gradients},
  author={Petersen, Brenden K and Larma, Mikel Landajuela and Mundhenk, T Nathan and Santiago, Claudio P and Kim, Soo K and Kim, Joanne T},
  journal={arXiv preprint arXiv:1912.04871},
  year={2019}
}
```
