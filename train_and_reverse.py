import gym
import classic_irl
from baselines.ppo2.ppo2 import


def train_ppo(env: gym.Env):
    print(dir(baselines.ppo2))


if __name__ == "__main__":
    train_ppo(gym.make("FrozenLake-v0"))

