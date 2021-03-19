import argparse
import os
import random

import numpy

import habitat
from habitat.core.simulator import Observations, Simulator
from typing import Any, Dict


class RandomAgent(habitat.Agent):
    def __init__(self, task_config: habitat.Config):
        self._POSSIBLE_ACTIONS = task_config.TASK.POSSIBLE_ACTIONS

    def reset(self):
        pass

    def act(self, observations):
        return {"action": numpy.random.choice(self._POSSIBLE_ACTIONS)}


from typing import Optional, Type
from habitat.config.default import get_config
import habitat
from habitat.datasets import make_dataset
from habitat import Config, Dataset
from gym.spaces import Box, Discrete
from gym.envs.registration import register
import numpy as np


class NavRLEnv(habitat.RLEnv):
    def __init__(self, config):
        config.defrost()
        self._rl_config = config.RL
        self._core_env_config = config
        self._reward_measure_name = self._rl_config.REWARD_MEASURE
        self._success_measure_name = self._rl_config.SUCCESS_MEASURE

        self._previous_measure = None
        self._previous_action = None

        # Config the dataset with given scene and object_category
        # dataset_config = config.DATASET
        # dataset_config.DATA_PATH = "data/datasets/train/content/%s.json.gz" % (
        # scene_id
        # )
        # dataset = habitat.make_dataset(
        # id_dataset=dataset_config.TYPE, config=dataset_config
        # )
        # dataset.episodes = [
        # ep
        # for ep in dataset.episodes
        # if ep.object_category == object_category
        # ]
        config.freeze()

        super().__init__(config)
        # self.observation_space = Box(
        # low=0,
        # high=255,
        # shape=(
        # self._core_env_config.SIMULATOR.RGB_SENSOR.WIDTH,
        # self._core_env_config.SIMULATOR.RGB_SENSOR.HEIGHT,
        # 3,
        # ),
        # dtype=np.uint8,
        # )
        # self.action_space = Discrete(4)

    def reset(self):
        self._previous_action = None
        observations = super().reset()
        self._previous_measure = self._env.get_metrics()[
            self._reward_measure_name
        ]
        return observations["rgb"]

    def step(self, *args, **kwargs):
        self._previous_action = args[0]
        observations, reward, done, info = super().step(*args, **kwargs)
        return observations["rgb"], reward, done, info

    def get_reward_range(self):
        return (
            self._rl_config.SLACK_REWARD - 1.0,
            self._rl_config.SUCCESS_REWARD + 1.0,
        )

    def get_reward(self, observations):
        reward = self._rl_config.SLACK_REWARD

        current_measure = self._env.get_metrics()[self._reward_measure_name]

        reward += self._previous_measure - current_measure
        self._previous_measure = current_measure

        if self._episode_success():
            reward += self._rl_config.SUCCESS_REWARD

        return reward

    def _episode_success(self):
        return self._env.get_metrics()[self._success_measure_name]

    def get_done(self, observations):
        done = False
        if self._env.episode_over or self._episode_success():
            done = True
        return done

    def get_info(self, observations):
        info = self.habitat_env.get_metrics()
        info["success"] = bool(self._episode_success())
        info["position"] = observations["gps"]
        info["heading"] = observations["compass"]
        return info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evaluation", type=str, required=True, choices=["local", "remote"]
    )
    args = parser.parse_args()

    config_paths = os.environ["CHALLENGE_CONFIG_FILE"]
    config = habitat.get_config(config_paths)
    agent = RandomAgent(task_config=config)
    env = NavRLEnv(config)
    obs = env.reset()
    breakpoint()
    action = agent.act(obs)
    x = env.step(action)

    if args.evaluation == "local":
        challenge = habitat.Challenge(eval_remote=False)
    else:
        challenge = habitat.Challenge(eval_remote=True)

    challenge.submit(agent)


if __name__ == "__main__":
    main()
