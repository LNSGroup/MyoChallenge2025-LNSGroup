# Run the baselines
This year we provide several baselines:
* Manipulation track with [SB3](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html) and curriculum learning. 
* A reflex-based locomotion controller, see [here](https://myosuite.readthedocs.io/en/latest/baselines.html#myolegreflex-baseline).

These baselines will not give you good task performance or win the challenge for you, but they provide a nice starting point.

To run the sb3-baselines with hydra, you need to install:

``` bash
pip install stable-baselines3[extra]
pip install hydra-core==1.1.0 hydra-submitit-launcher submitit
#optional
pip install tensorboard wandb
```
Take a look [here](https://stable-baselines3.readthedocs.io/en/master/guide/install.html) if you run into issues.
The requirements for the reflex-based baseline are contained in the above link.

## Table Tennis Track
This sb3-baseline allow the torso to stand up temporarily and hit the table tennis ball with ~ 5% success rate.

A complete tutorial can be found here with a downloadable baseline [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1jQFmID4mo7KnlpngMiuY98iYGmylQ3IZ?usp=sharing)

## Soccer track
This deprl-baseline will try to stand around and slowly move across the quad.
``` python
import gym
import myosuite, deprl

env = gym.make('myoChallengeRunTrackP2-v0')
policy = deprl.load_baseline(env)

for ep in range(5):
    print(f'Episode: {ep} of 5')
    state = env.reset()
    while True:
        action = policy(state)
        # uncomment if you want to render the task
        # env.mj_render()
        next_state, reward, done, info = env.step(action)
        state = next_state
        if done: 
            break
```


