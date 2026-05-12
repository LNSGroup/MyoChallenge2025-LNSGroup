import os
import sys
import gymnasium as gym
import numpy as np
import collections
import time
from typing import Dict, Any, Optional, List
import enum
import mujoco
from scipy.spatial.transform import Rotation as R


# total number of observations: 414
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

DEFAULT_RWD_KEYS_AND_WEIGHTS = {
        "paddle_quat": 1,
        'torso_up': 1,
        'palm_dist': 1,
        "racket_tracking":2,
        "sparse":10,
        "good_sparse":100,
        "own_side_penalty":5
    }


class tabletennisWrapper(gym.Wrapper):
    """
    Gym wrapper for TableTennisEnvV0 with:
    - Manual curriculum stage control
    - Custom reward computation and weighted sum
    - Compatible rendering and observation
    """
    metadata: Dict[str, Any] = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 50,
    }

    def __init__(self, env, reward_dict=None, curriculum_stage=0):
        super().__init__(env)
        #self.reward_dict = reward_dict or env.DEFAULT_RWD_KEYS_AND_WEIGHTS.copy()
        self.reward_dict = DEFAULT_RWD_KEYS_AND_WEIGHTS
        self.sim_step = 0
        self.curriculum_stage = curriculum_stage  # manual curriculum stage index
        self.reward_items = collections.OrderedDict()
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(273, ), dtype=np.float32)  
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(431, ), dtype=np.float32)
        # Reward keys enabled per curriculum stage
        self.stage_reward_keys = {
            0:["paddle_quat", "torso_up","palm_dist", "racket_tracking","sparse", "good_sparse","own_side_penalty"], 
            1: ["act_reg", "paddle_quat", "torso_up","palm_dist", "sparse","racket_tracking"]
        }
        self.ball_touch_counter = 0
        self.data_log = []   # episode-level logs (e.g. landing xy)

        timestep = self.model.opt.timestep
        self.metadata["render_fps"] = int(round(1.0 / timestep / self.frame_skip))

    def __getattr__(self, name):
        if hasattr(self.env, name):
            return getattr(self.env, name)
        return getattr(self.env.sim, name)

    def render(self, mode=None):
        if self.render_mode == 'human':
            self.env.mj_render()
        elif self.render_mode == 'rgb_array':
            frame_size = (400, 400)
            return self.env.sim.renderer.render_offscreen(
                frame_size[0], frame_size[1], camera_id=4, device_id=0
            )

    def compute_reward(self, obs_dict, action):
        rwd_items = collections.OrderedDict()
        keys = self.stage_reward_keys.get(self.curriculum_stage, list(self.reward_dict.keys()))

        for key in keys:
            if key == "reach_dist":
                err = np.linalg.norm(obs_dict['reach_err'])
                rwd_items[key] = np.exp(-2. * err)
            elif key == "paddle_quat":
                t_sim = obs_dict['time']
                quat = obs_dict['paddle_ori'][[1, 2, 3, 0]]  # env layout -> scipy (x,y,z,w)
                paddle_rot = R.from_quat(quat).as_matrix()
                paddle_dir = paddle_rot[:, 2]
                paddle_dir_norm = paddle_dir / (np.linalg.norm(paddle_dir) + 1e-8)

                # Racket face normal vs predicted swing direction
                v_racket_norm = self._v_racket / (np.linalg.norm(self._v_racket) + 1e-8)

                cos_hit = np.abs(np.dot(paddle_dir_norm, v_racket_norm))
                sparse_rwd = (cos_hit + 1) / 2  # in [0, 1]
                rwd_items["paddle_quat"] = sparse_rwd
            elif key == "palm_dist":
                err = np.linalg.norm(obs_dict['palm_handle_err'])
                rwd_items[key] = np.exp(-5. * err)
            elif key == "torso_up":
                torso_err = self.stand_straight()
                rwd_items[key] = np.exp(-5. * torso_err)
                #rwd_items["torso_up"] = 1.0 / (1.0 + 5 * torso_err)
            elif key == "act_reg":
                act_mag = np.linalg.norm(action) / len(action)
                rwd_items[key] = -1. * act_mag
            elif key == "sparse":
                paddle_touch = (
                    obs_dict['touching_info'][0][0]
                    if obs_dict['touching_info'].ndim == 3
                    else obs_dict['touching_info']
                )
                rwd_items[key] = float(paddle_touch[0] == 1)
            elif key == "solved":
                solved = self.env.evaluate_pingpong_trajectory(self.env.contact_trajectory) is None
                rwd_items[key] = float(solved)
                
            elif key == "racket_tracking":
                pos_err = np.linalg.norm(obs_dict['paddle_pos'] - self._hit_pos)
                rwd_items[key] = np.exp(-5 * pos_err)
                
            elif key == "good_sparse":
                paddle_touch = (
                    obs_dict['touching_info'][0][0]
                    if obs_dict['touching_info'].ndim == 3
                    else obs_dict['touching_info']
                )

                if paddle_touch[2] == 1:
                    ball_xy = obs_dict["ball_pos"][:2]

                    # Opponent-side landing band (table half-plane heuristic)
                    if (-1.37/2 - 0.3 <= ball_xy[0] <= -1.37/2 + 0.3) and (0.04-0.35 <= ball_xy[1] <=0.04+0.35):
                        rwd_items[key] = 1.5
                    else:
                        rwd_items[key] = 1
                else:
                    rwd_items[key] = 0.0
                
            elif key == "own_side_penalty":
                ball_touch = (
                    obs_dict['touching_info'][0][0]
                    if obs_dict['touching_info'].ndim == 3
                    else obs_dict['touching_info']
                )
                if ball_touch[1] == 1 and obs_dict["time"] > self._t_hit:
                    # Penalize ball landing on own half after predicted hit time
                    rwd_items["own_side_penalty"] = -1.0
                else:
                    rwd_items["own_side_penalty"] = 0.0
                    
            else:
                rwd_items[key] = 0.0

        total_reward = sum(rwd_items[k] * self.reward_dict.get(k, 1.0) for k in rwd_items)
        return total_reward, rwd_items
    
    
    def get_custom_obs(self, obs_vec):
        # Concat policy-side extras: predicted hit pos/vel/time
        extra = np.concatenate([
            self._hit_pos,         # (3,)
            self._v_racket,        # (3,)
            np.array([self._t_hit])  # (1,)
        ])
        return np.concatenate([obs_vec, extra])

    def reset(self, **kwargs):
        obs, temp_info = self.env.reset(**kwargs)
        self.sim_step = 0
        self.ball_touch_counter = 0

        obs_dict = self.env.get_obs_dict(self.env.sim)
        try:
            self._hit_pos, self._v_racket, self._v0, self._t_hit, _ = self.predict_racket_motion(
                pos_init=obs_dict['ball_pos'],
                vel_init=obs_dict['ball_vel']
            )
        except:
            self._hit_pos, self._v_racket, self._v0, self._t_hit, _ = None, None, None, None, None

        # Per-episode diagnostic log (initialized each reset)
        self.current_log = {
            "pos_init": obs_dict['ball_pos'].copy(),
            "vel_init": obs_dict['ball_vel'].copy(),
            "target_hit_pos": self._hit_pos.copy() if self._hit_pos is not None else np.zeros(3),
            "v_racket": self._v_racket.copy() if self._v_racket is not None else np.zeros(3),
            "final_xy": None
        }

        obs_custom = self.get_custom_obs(obs)
        self.target_hip_x = 0
        self.target_hip_y = -self._hit_pos[1] + 0.5 if self._hit_pos is not None else 0

        return obs_custom, temp_info

    def step(self, action):
        full_action = np.zeros(275, dtype=np.float32)
        full_action[:273] = action
        full_action[273] = self.target_hip_x
        full_action[274] = self.target_hip_y

        obs, base_reward, terminated, truncated, info = self.env.step(full_action)
        obs_dict = self.env.get_obs_dict(self.env.sim)
        # Record first table crossing (approximate bounce xy)
        if self.current_log["final_xy"] is None:
            ball_z = obs_dict['ball_pos'][2]
            table_z = 0.82  # table surface height in this env
            if ball_z <= table_z and obs_dict["time"] > 0.6:
                self.current_log["final_xy"] = obs_dict['ball_pos'][:2].copy()  # x, y

        total_reward, reward_items = self.compute_reward(obs_dict, full_action)
        info_train = {"total_reward": total_reward}
        for k, v in reward_items.items():
            info_train[f"reward/{k}"] = v

        obs_custom = self.get_custom_obs(obs)
        done = terminated or truncated

        # On episode end, append log snapshot
        if done:
            if not hasattr(self, "data_log"):
                self.data_log = []
            self.data_log.append(self.current_log)
        if terminated or truncated:
            # Success flag from contact trajectory rules
            traj_issue = evaluate_pingpong_trajectory(self.env.contact_trajectory)
            if traj_issue is None:
                info_train["success"] = True
            else:
                info_train["success"] = False
        return obs_custom, total_reward, terminated, truncated, info_train
    
    def stand_straight(self):
        neutral_angles = {
            "flex_extension": 0.0,
            "lat_bending": 0.0,
            "Abs_t1": 0.0,
            "Abs_t2": 0.0,
            "Abs_r3": 0.0,
            "L4_L5_FE": 0.0,
            "L4_L5_LB": 0.0,
            "L3_L4_FE": -0.0972,   # nominal reference angle
            "L3_L4_LB": 0.0,
            "L2_L3_FE": 0.0,
            "L2_L3_LB": 0.0,
        }

        # Aggregate joint deviation from nominal standing pose
        errs = []
        for jname, target in neutral_angles.items():
            q = self.sim.data.qpos[
                self.sim.model.jnt_qposadr[self.sim.model.joint_name2id(jname)]
            ]
            errs.append(q - target)   # deviation from nominal

        torso_err = np.linalg.norm(errs) 
        return torso_err
        
    
    def predict_racket_motion(self, pos_init, vel_init, 
                          pl=np.array([-1.37/2, 0.04, 0.79847851]), t0=0.37, 
                          g=9.81, e=0.97701286, contact_z=0.79847851, 
                          dt=1e-3, t_max=1.0, k=0, hit_plane_x=1.9, cr=1.0):
        """
        Predict racket contact and post-impact ball velocity.

        Args:
            pos_init: Initial ball position (3,).
            vel_init: Initial ball velocity (3,).
            pl: Target landing point on the far table.
            t0: Flight time from racket contact to target.
            g: Gravity.
            e: Table restitution.
            contact_z: Table height (z).
            dt, t_max: Integrator step and horizon.
            k: Quadratic air drag (0 disables).
            hit_plane_x: Racket contact plane x.
            cr: Ball-racket restitution.

        Returns:
            hit_pos, v_racket, v0, t_hit, t_racket
        """

        pos = pos_init.copy()
        vel = vel_init.copy()
        t = 0.0
        bounced = False
        hit_pos = None
        hit_vel = None
        t_hit = None

        while t < t_max:
            speed = np.linalg.norm(vel)
            acc = np.array([0,0,-g]) - k * speed * vel
            vel += acc * dt
            pos += vel * dt
            t += dt

            # Table bounce
            if not bounced and pos[2] <= contact_z and vel[2] < 0:
                vel[2] = -e * vel[2]
                pos[2] = contact_z
                bounced = True

            # Intersect forward hit plane
            if hit_pos is None and pos[0] >= hit_plane_x:
                prev_pos = pos - vel * dt
                prev_vel = vel - acc * dt
                alpha = (hit_plane_x - prev_pos[0]) / (pos[0] - prev_pos[0])
                hit_pos = prev_pos + alpha * (pos - prev_pos)
                hit_vel = prev_vel + alpha * (vel - prev_vel)
                t_hit = t - dt + alpha * dt
                break

        if hit_pos is None:
            raise ValueError("Ball did not reach hit plane (x >= hit_plane_x) within t_max")

        # Post-impact target ball velocity
        g0 = np.array([0,0,g])
        v0 = (pl - hit_pos) / t0 + 0.5 * g0 * t0

        # Required racket velocity for elastic collision toward v0
        diff = v0 - hit_vel
        u = diff / np.linalg.norm(diff)
        v_racket = ((np.dot(v0, u) + cr * np.dot(hit_vel, u)) / (1 + cr)) * u

        t_racket = t_hit

        return hit_pos, v_racket, v0, t_hit, t_racket
    

  
class IdInfo:
    def __init__(self, model: mujoco.MjModel):
        self.paddle_sid = model.site("paddle").id
        self.paddle_bid = model.body("paddle").id
        self.ball_sid = model.site("pingpong").id
        self.ball_bid = model.body("pingpong").id
        self.handle_sid = model.site("handle_site").id

        self.ball_bid = model.body("pingpong").id
        self.own_half_gid = model.geom("coll_own_half").id
        self.paddle_gid = model.geom("pad").id
        self.opponent_half_gid = model.geom("coll_opponent_half").id
        self.ground_gid = model.geom("ground").id
        self.net_gid = model.geom("coll_net").id

        myo_bodies = [model.body(i).id for i in range(model.nbody)
                      if not model.body(i).name.startswith("ping")
                      and "paddle" not in model.body(i).name
                      and not model.body(i).name in ["pingpong"]]
        self.myo_body_range = (min(myo_bodies), max(myo_bodies))

        # TODO add locomotion joint ids

        self.myo_joint_range = np.concatenate([model.joint(i).qposadr for i in range(model.njnt)
                                            if not model.joint(i).name.startswith("ping")
                                            and not model.joint(i).name == "pingpong_freejoint"
                                            and not model.joint(i).name == "paddle_freejoint"])

        self.myo_dof_range = np.concatenate([model.joint(i).dofadr for i in range(model.njnt)
                                            if not model.joint(i).name.startswith("ping")
                                            and not model.joint(i).name == "paddle_freejoint"])
          
        

    

class PingpongContactLabels(enum.Enum):
    PADDLE = 0 # TODO: Remove collisions with myo
    OWN = 1
    OPPONENT = 2
    GROUND = 3
    NET = 4
    ENV = 5


class ContactTrajIssue(enum.Enum):
    OWN_HALF = 0
    MISS = 1
    NO_PADDLE = 2
    DOUBLE_TOUCH = 3


def get_ball_contact_labels(model: mujoco.MjModel, data: mujoco.MjData, id_info: IdInfo):
    for con in data.contact:
        if model.geom(con.geom1).bodyid == id_info.ball_bid:
            yield geom_id_to_label(con.geom2, id_info)
        elif model.geom(con.geom2).bodyid == id_info.ball_bid:
            yield geom_id_to_label(con.geom1, id_info)



def geom_id_to_label(body_id, id_info: IdInfo):
    if body_id == id_info.paddle_gid:
        return PingpongContactLabels.PADDLE
    elif body_id == id_info.own_half_gid:
        return PingpongContactLabels.OWN
    elif body_id == id_info.opponent_half_gid:
        return PingpongContactLabels.OPPONENT
    elif body_id == id_info.net_gid:
        return PingpongContactLabels.NET
    elif body_id == id_info.ground_gid:
        return PingpongContactLabels.GROUND
    else:
        return PingpongContactLabels.ENV

def evaluate_pingpong_trajectory(contact_trajectory: list[set]):
    has_hit_paddle = False
    has_bounced_from_paddle = False
    has_bounced_from_table = False
    own_contact_count = 0
    own_contact_phase_done = False

    for s in contact_trajectory:
        own_in_s = any(elem.value == PingpongContactLabels.OWN.value for elem in s)
        paddle_in_s = any(elem.value == PingpongContactLabels.PADDLE.value for elem in s)
        opponent_in_s = any(elem.value == PingpongContactLabels.OPPONENT.value for elem in s)

        if paddle_in_s:
            has_hit_paddle = True

        if own_in_s:
            if not has_bounced_from_table:
                # Start of initial bounce from serving
                has_bounced_from_table = True
                own_contact_count = 1
            elif not own_contact_phase_done:
                own_contact_count += 1
                if own_contact_count > 4:  # initial serving bounce has contact for 2 timesteps
                    own_contact_phase_done = True
                    return ContactTrajIssue.OWN_HALF
                
            else:
                return ContactTrajIssue.OWN_HALF
        elif has_bounced_from_table:
            # Exit the initial own bounce phase
            own_contact_phase_done = True

        if opponent_in_s:
            if has_hit_paddle:
                return None
            else:
                return ContactTrajIssue.NO_PADDLE
    return ContactTrajIssue.MISS


