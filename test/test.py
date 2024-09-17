from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.agents.agent import NNPolicy, AgentFromPolicy, AgentPair
from overcooked_ai_py.agents.benchmarking import AgentEvaluator
from overcooked_ai_py.visualization.state_visualizer import StateVisualizer
from overcooked_ai_py.agents.benchmarking import AgentEvaluator
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
import os
import json

layout = "coordination_ring"
with open('../training_data/' + layout + ".json", 'r') as openfile:
    episode_soups = json.load(openfile)
print(episode_soups)


def test_one():
    GYM_ID = "CartPole-v1"
    env = gym.make(GYM_ID)
    obs, _ = env.reset()
    n_features = len(obs)
    n_actions = env.action_space.n


    model = nn.Sequential(
            nn.Linear(in_features=n_features,
                    out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32,
                    out_features=n_actions))

    obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
    logits = model(obs_tensor)
    print(logits)
    cat = Categorical(logits=logits)
    action_zero = 0
    action_one = 0
    n = 100000
    for i in range(n):
        temp = cat.sample()
        action = temp.item()
        if (action == 0):
            action_zero += 1
        else:
            action_one += 1
    print("----------------------")
    print(action_zero/n)
    print(action_one/n)
