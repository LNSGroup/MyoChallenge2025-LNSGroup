import os
import sys
import gymnasium as gym
import myosuite

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from challenge_wrapper import *

ALL_PINGPONG_OBS_KEYS = [
    'time',             # 1 
    'pelvis_pos',       # 3
    'body_qpos',        # 58
    'body_qvel',        # 58
    'ball_pos',         # 3
    'ball_vel',         # 3
    'paddle_pos',       # 3
    'paddle_vel',       # 3
    'reach_err',        # 3
    'touching_info',    # 6
    'act',              # 273
]

ALL_SOCCER_OBS_KEYS = [
    'time',             # 1 
    'internal_qpos',    # 46
    'internal_qvel',    # 46
    'grf',              # 4
    'torso_angle',      # 4
    'pelvis_angle',     # 4
    'muscle_length',    # 290
    'muscle_velocity',  # 290
    'muscle_force',     # 290
    'r_toe_pos',        # 3
    'l_toe_pos',        # 3
    'act',              # 290
    'ball_pos',         # 3
    'goal_bounds',      # 12
    'model_root_pos',   # 7
    'model_root_vel',   # 6
]

# env = gym.make('myoChallengeTableTennisP1-v0', obs_keys=ALL_PINGPONG_OBS_KEYS)         # Pingpong
# env = PingpongWrapper(env)
env = gym.make('myoChallengeSoccerP2-v0', obs_keys=ALL_SOCCER_OBS_KEYS)         # Soccer
env = SoccerTurnKickWrapper(env, random_init=True, imitation_reset_type='kick', replay=False, rotate_to_mid=True, motion_list=[0, 1])
# env = SoccerImitationMoveWrapper(env, random_init=True, replay=True, start_hold_time=0.5, end_hold_time=0.5, traj_time=3.0)
# env = SoccerImitationWalkWrapper(env, random_init=True, imitation_reset_type='keyframe', replay=True)
# env = SoccerImitationKickWrapper(env, random_init=True, imitation_reset_type='keyframe', replay=False, motion_list=[14, 15])

episodes = 20
for ep in range(episodes):
    print(f'Episode: {ep} of {episodes}')
    obs, _ = env.reset()
    action = env.action_space.sample()
    env.mj_render()
    print("action shape: ", action.shape)
    print("obs shape: ", obs.shape)
    
    while True:
        action = env.action_space.sample()  * 0.
        # uncomment if you want to render the task
        env.mj_render()
        next_state, reward, terminated, truncated, info = env.step(action)
        # print("reward: ", reward)
        # print("info: ", info)
        state = next_state 
        if terminated or truncated: 
            break