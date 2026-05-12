import grpc
import gymnasium as gym
import pickle
import sys
import os
import time
from concurrent import futures

import evaluation_pb2
import evaluation_pb2_grpc
import numpy as np
LOCAL_EVALUATION = os.environ.get("LOCAL_EVALUATION")
EVALUATION_COMPLETED = False

import myosuite

# -----------------------------
# Environment wrapper with trajectory recording
# -----------------------------
class evaluator_environment:
    def __init__(self, environment="myoChallengeTableTennisP2-v0"):
        self.score = 0
        self.feedback = None
        self.environment = environment
        self.env = gym.make(environment)
        self.trajectory = []  # Store per-step metrics
        # Set initial seed for reproducibility
        self.env.seed(42) if hasattr(self.env, 'seed') else None

    def get_output_keys(self):
        print(self.env.obs_keys)
        return self.env.obs_keys

    def set_output_keys(self, key_set):
        self.env = gym.make(self.environment, obs_keys=key_set)
        # Re-set seed after recreating environment
        self.env.seed(42) if hasattr(self.env, 'seed') else None

    def reset(self, seed=None):
        # IMPORTANT: Call env.seed() before env.reset() to ensure reproducibility
        # The reset(seed=...) parameter alone doesn't properly seed the environment's np_random
        if seed is not None:
            self.env.seed(seed)
            obs = self.env.reset()
        else:
            obs = self.env.reset()
        self.feedback = None
        self.trajectory = []
        return obs

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated

        # Record solved, act_reg, reward
        solved = info.get('rwd_dict', {}).get('solved', 0)
        act_reg = info.get('rwd_dict', {}).get('act_reg', 0)
        self.trajectory.append({'solved': solved, 'act_reg': act_reg, 'reward': reward})

        self.feedback = [obs, reward, done]
        return obs, reward, done, info

    def get_action_space(self):
        return len(self.env.action_space.sample())

    def get_observation_space(self):
        return len(self.env.observation_space.sample())

    def get_obsdict(self):
        return self.env.get_obs_dict(self.env.sim)

    def next_score(self):
        self.score += 1

    def compute_metrics(self):
        """
        Compute success rate and effort based on trajectory
        """
        path = self.trajectory
        num_success = int(np.sum([s['solved'] for s in path]) >= 1)
        score = num_success
        effort = -1.0 * np.mean([s['act_reg'] for s in path]) if path else 0
        total_reward = np.sum([s['reward'] for s in path])
        return {'score': score, 'effort': effort, 'total_reward': total_reward}


# -----------------------------
# gRPC Service
# -----------------------------
class Environment(evaluation_pb2_grpc.EnvironmentServicer):
    def __init__(self, challenge_pk, phase_pk, submission_pk, server):
        self.challenge_pk = challenge_pk
        self.phase_pk = phase_pk
        self.submission_pk = submission_pk
        self.server = server
        self.iter = 0
        self.repetition = 0

    def set_output_keys(self, request, context):
        new_out_keys = unpack_for_grpc(request.SerializedEntity)
        message = pack_for_grpc(env.set_output_keys(new_out_keys))
        return evaluation_pb2.Package(SerializedEntity=message)

    def reset(self, request, context):
        self.iter = 0
        self.repetition += 1
        reset_request = unpack_for_grpc(request.SerializedEntity)
        seed = reset_request if isinstance(reset_request, int) else None
        message = pack_for_grpc(env.reset(seed=seed))
        env.feedback = []
        return evaluation_pb2.Package(SerializedEntity=message)

    def get_action_space(self, request, context):
        message = pack_for_grpc(env.get_action_space())
        return evaluation_pb2.Package(SerializedEntity=message)

    def get_observation_space(self, request, context):
        message = pack_for_grpc(env.get_observation_space())
        return evaluation_pb2.Package(SerializedEntity=message)

    def get_obsdict(self, request, context):
        message = pack_for_grpc(env.get_obsdict())
        return evaluation_pb2.Package(SerializedEntity=message)

    def act_on_environment(self, request, context):
        global EVALUATION_COMPLETED
        if not env.feedback or not env.feedback[2]:
            action = unpack_for_grpc(request.SerializedEntity)
            env.next_score()
            env.step(action)

        feedback = [env.feedback[0], env.feedback[1], env.feedback[2]]
        if self.repetition == 101:
            EVALUATION_COMPLETED = True
            metrics = env.compute_metrics()
            print("\n=== FINAL SUCCESS RATE METRICS ===")
            print(metrics)
        self.iter += 1

        rwd_dict = env.trajectory[-1] if env.trajectory else {}
        return evaluation_pb2.Package(
            SerializedEntity=pack_for_grpc({
                "feedback": feedback,
                "current_score": env.score,
                "eval_completed": EVALUATION_COMPLETED,
                "rwd_dict": rwd_dict,
            })
        )


# -----------------------------
# Global env
# -----------------------------
env = evaluator_environment()


# -----------------------------
# Serialization helpers
# -----------------------------
def pack_for_grpc(entity):
    return pickle.dumps(entity)

def unpack_for_grpc(entity):
    return pickle.loads(entity)


# -----------------------------
# Main server loop
# -----------------------------
def main():
    challenge_pk = "1"
    phase_pk = "1"
    submission_pk = "1"

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    evaluation_pb2_grpc.add_EnvironmentServicer_to_server(
        Environment(challenge_pk, phase_pk, submission_pk, server), server
    )

    print("Starting gRPC server on port 8085...")
    server.add_insecure_port("[::]:8085")
    server.start()

    try:
        while not EVALUATION_COMPLETED:
            time.sleep(2)
        server.stop(0)
    except KeyboardInterrupt:
        server.stop(0)

    exit(0)


if __name__ == "__main__":
    main()