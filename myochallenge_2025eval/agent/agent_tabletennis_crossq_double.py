#!/usr/bin/env python3
import os
import pickle
import time
import numpy as np
import argparse
import json
import joblib

import evaluation_pb2
import evaluation_pb2_grpc
import grpc
from utils import RemoteConnection
from crossq_agent import load_tabletennis


# Ball trajectory and racket velocity prediction (physics rollout).
def predict_racket_motion(pos_init, vel_init,
                          pl=np.array([-1.37 / 2, 0.04, 0.79847851]), t0=0.37,
                          g=9.81, e=0.97701286, contact_z=0.79847851,
                          dt=1e-3, t_max=1.4, k=0, hit_plane_x=1.9, cr=1.0):
    pos = pos_init.copy()
    vel = vel_init.copy()
    t = 0.0
    bounced = False
    bounce_pos = None
    hit_pos = None
    hit_vel = None
    t_hit = None

    while t < t_max:
        speed = np.linalg.norm(vel)
        acc = np.array([0, 0, -g]) - k * speed * vel
        vel += acc * dt
        pos += vel * dt
        t += dt

        # First table bounce
        if not bounced and pos[2] <= contact_z and vel[2] < 0:
            prev_pos = pos - vel * dt
            denom = pos[2] - prev_pos[2]
            alpha = (contact_z - prev_pos[2]) / denom if denom != 0 else 0
            bounce_pos = prev_pos + alpha * (pos - prev_pos)
            vel[2] = -e * vel[2]
            pos[2] = contact_z
            bounced = True

        # Hit plane crossing
        if hit_pos is None and pos[0] >= hit_plane_x:
            prev_pos = pos - vel * dt
            prev_vel = vel - acc * dt
            denom = pos[0] - prev_pos[0]
            alpha = (hit_plane_x - prev_pos[0]) / denom if denom != 0 else 0
            hit_pos = prev_pos + alpha * (pos - prev_pos)
            hit_vel = prev_vel + alpha * (vel - prev_vel)
            t_hit = t - dt + alpha * dt
            break

    if hit_pos is None:
        hit_pos, hit_vel, t_hit = np.zeros(3), np.zeros(3), 0.0
    if bounce_pos is None:
        bounce_pos = np.zeros(3)

    g0 = np.array([0, 0, g])
    v0 = (pl - hit_pos) / t0 + 0.5 * g0 * t0
    diff = v0 - hit_vel
    norm_diff = np.linalg.norm(diff)
    if norm_diff == 0:
        u = np.array([1.0, 0, 0])
    else:
        u = diff / norm_diff
    v_racket = ((np.dot(v0, u) + cr * np.dot(hit_vel, u)) / (1 + cr)) * u
    return hit_pos, v_racket, v0, t_hit, bounce_pos


# Build policy observation vector from remote obs + predicted hit features.
def get_custom_observation(rc, obs_keys, _hit_pos, _v_racket, _t_hit):
    obs_dict = rc.get_obsdict()
    raw_obs = rc.obsdict2obsvec(obs_dict, obs_keys)
    extra_obs = np.concatenate([_hit_pos, _v_racket, np.array([_t_hit])])
    return np.concatenate([raw_obs, extra_obs])


# Six CrossQ policies, GPC routing, shared hit-plane groups per plane_x.
time.sleep(10)
LOCAL_EVALUATION = os.environ.get("LOCAL_EVALUATION")

base_dir = os.path.dirname(__file__)
print("Loading agents...")

AGENTS = ["A", "B", "C", "D", "E", "F"]
policies = {aid: load_tabletennis(os.path.join(base_dir, f"checkpoint_agent_{aid}")) for aid in AGENTS}
print("Agents loaded successfully.")

# Load GPC classifiers (hit yz -> success probability).
gpc_dir = os.path.join(base_dir, "gpc")
gpc_A = joblib.load(os.path.join(gpc_dir, "extreme_right_gpc_model.joblib"))
gpc_B = joblib.load(os.path.join(gpc_dir, "extreme_left_gpc_model.joblib"))
gpc_C = joblib.load(os.path.join(gpc_dir, "right_gpc_model.joblib"))
gpc_D = joblib.load(os.path.join(gpc_dir, "left_gpc_model.joblib"))
gpc_E = joblib.load(os.path.join(gpc_dir, "add_right1_gpc_model.joblib"))
gpc_F = joblib.load(os.path.join(gpc_dir, "add_left_gpc_model.joblib"))

print("GPC models loaded successfully.")

# Policy obs keys from checkpoint A config.
config = json.load(open(os.path.join(base_dir, "checkpoint_agent_A", 'PingPongv2.json')))
arg_config = argparse.Namespace(**config)
custom_obs_keys = arg_config.single_env_kwargs['obs_keys']

# gRPC bridge to evaluation environment
rc = RemoteConnection("environment:8085" if LOCAL_EVALUATION else "localhost:8085")
rc.set_output_keys(custom_obs_keys)

# Per-agent: policy, GPC model, hit plane, hip targets
agent_configs = {
    "A": dict(policy=policies["A"], gpc=gpc_A, hit_plane_x=1.4, target_hip_x=1.0, target_hip_y=0.5),
    "B": dict(policy=policies["B"], gpc=gpc_B, hit_plane_x=1.4, target_hip_x=1.0, target_hip_y=0.0),
    "C": dict(policy=policies["C"], gpc=gpc_C, hit_plane_x=1.65, target_hip_x=0.4, target_hip_y=0.5),
    "D": dict(policy=policies["D"], gpc=gpc_D, hit_plane_x=1.65, target_hip_x=0.4, target_hip_y=-0.1),
    "E": dict(policy=policies["E"], gpc=gpc_E, hit_plane_x=1.9, target_hip_x=0, target_hip_y=0.5),
    "F": dict(policy=policies["F"], gpc=gpc_F, hit_plane_x=1.65, target_hip_x=0.4, target_hip_y=-0.1),

}

# Agents sharing the same plane_x reuse one physics prediction.
plane_groups = {
    1.4: [ "B", "A"],
    1.65: ["C", "D", "F"],
    1.9: ["E"],
}

# ==============================================================
# Evaluation Loop
# ==============================================================
flat_completed = None
trial = 0

while not flat_completed:
    print(f"\n=== Trial {trial} ===")
    rc.reset()
    obs_dict = rc.get_obsdict()

    pos_init = obs_dict['ball_pos']
    vel_init = obs_dict['ball_vel']

    # Step 1: predict per shared hit plane
    hit_results = {}
    for plane_x, aids in plane_groups.items():
        hit_pos, v_racket, v0, t_hit, bounce_pos = predict_racket_motion(
            pos_init, vel_init, hit_plane_x=plane_x
        )
        for aid in aids:
            hit_results[aid] = dict(hit_pos=hit_pos, v_racket=v_racket, t_hit=t_hit)

    # Step 2: GPC success probability per agent
    probs = {}
    for key, cfg in agent_configs.items():
        hit_yz = hit_results[key]['hit_pos'][1:3].reshape(1, -1)
        probs[key] = cfg['gpc'].predict_proba(hit_yz)[0, 1]
    print("GPC probs:", probs)

    # Step 3: pick agent with highest GPC probability
    best_agent = max(probs, key=probs.get)
    active_cfg = agent_configs[best_agent]
    _hit_pos = hit_results[best_agent]['hit_pos']
    _v_racket = hit_results[best_agent]['v_racket']
    _t_hit = hit_results[best_agent]['t_hit']
    print(f"Chosen Agent: {best_agent}")

    target_hip_x = active_cfg['target_hip_x']
    target_hip_y = active_cfg['target_hip_y'] - _hit_pos[1]
    active_policy = active_cfg['policy']

    # Step 4: rollout with selected policy
    ret = 0
    flag_trial = None
    while not flag_trial:
        obs = get_custom_observation(rc, custom_obs_keys, _hit_pos, _v_racket, _t_hit)
        policy_action = active_policy(obs)

        full_action = np.zeros(275, dtype=np.float32)
        full_action[:273] = policy_action
        full_action[273] = target_hip_x
        full_action[274] = target_hip_y

        base = rc.act_on_environment(full_action)
        obs = base["feedback"][0]
        flag_trial = base["feedback"][2]
        flat_completed = base["eval_completed"]
        ret += base["feedback"][1]

        if flag_trial:
            print(f"Return: {ret:.3f}")
            print("=" * 80)
            break

    trial += 1