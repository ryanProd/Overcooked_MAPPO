from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.agents.agent import NNPolicy, AgentFromPolicy, AgentPair
from overcooked_ai_py.agents.benchmarking import AgentEvaluator
from overcooked_ai_py.visualization.state_visualizer import StateVisualizer
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
import os

### Environment setup ###

## Swap between the 5 layouts here:
layout = "cramped_room"
# layout = "asymmetric_advantages"
# layout = "coordination_ring"
# layout = "forced_coordination"
# layout = "counter_circuit_o_1order"

## Reward shaping is disabled by default; i.e., only the sparse rewards are
## included in the reward returned by the enviornment).  If you'd like to do
## reward shaping (recommended to make the task much easier to solve), this
## data structure provides access to a built-in reward-shaping mechanism within
## the Overcooked environment.  You can, of course, do your own reward shaping
## in lieu of, or in addition to, using this structure. The shaped rewards
## provided by this structure will appear in a different place (see below)
reward_shaping = {
    "PLACEMENT_IN_POT_REW": 3,
    "DISH_PICKUP_REWARD": 3,
    "SOUP_PICKUP_REWARD": 5
}

# Length of Episodes.  Do not modify for your submission!
# Modification will result in a grading penalty!
horizon = 400

# Build the environment.  Do not modify!
mdp = OvercookedGridworld.from_layout_name(layout, rew_shaping_params=reward_shaping)
base_env = OvercookedEnv.from_mdp(mdp, horizon=horizon, info_level=0)
env = gym.make("Overcooked-v0", base_env=base_env, featurize_fn=base_env.featurize_state_mdp)

obs = env.reset()
print(type(obs["both_agent_obs"][0][0]))
### Train your agent ###

# drive.mount('/content/drive')
# !mkdir -p "/content/drive/My Drive/Colab"

# The code below runs a few episodes with a random agent.  Your learning algorithm
# would go here.

def random_run():

    num_episodes = 5

    for e in range(num_episodes):
        # Episode termination flag
        done = False

        # The number of soups the agent pair made during the episode
        num_soups_made = 0

        # Reset the environment at the start of each episode
        obs = env.reset()

        while not done:
            # Obtain observations for each agent
            obs0 = obs["both_agent_obs"][0]
            obs1 = obs["both_agent_obs"][1]

            # Select random actions from the set {North, South, East, West, Stay, Interact}
            # for each agent.
            a0 = env.action_space.sample()
            a1 = env.action_space.sample()

            # Take the selected actions and receive feedback from the environment
            # The returned reward "R" only reflects completed soups.
            obs, R, done, info = env.step([a0, a1])
            
            # You can find the separate shaping rewards induced by the data
            # structure you defined above in the "info" dictionary.
            ## THE REVERSAL OF THIS ARRAY IS NECESSARY TO ALIGN THE CORRECT REWARD
            ## TO THE CORRECT AGENT (see project documentation)!
            # Note that this shaping reward does *not* include the +20 reward for
            # completed soups (the one returned in "R").
            r_shaped = info["shaped_r_by_agent"]
            if env.agent_idx:
                r_shaped_0 = r_shaped[1]
                r_shaped_1 = r_shaped[0]
            else:
                r_shaped_0 = r_shaped[0]
                r_shaped_1 = r_shaped[1]

            # Accumulate the number of soups made
            num_soups_made += int(R / 20) # Each served soup generates 20 reward


        # Display status
        print("Ep {0}".format(e + 1), end=" ")
        print("shaped reward for agent 0: {0}:".format(r_shaped_0), end=" ")
        print("shaped reward for agent 1: {0}".format(r_shaped_1), end=" ")
        print("number of soups made: {0}".format(num_soups_made))

    # The info flag returned by the environemnt contains special status info
    # specifically when done == True.  This information may be useful in
    # developing, debugging, and analyzing your results.  It may also be a good
    # way for you to find a metric that you can use in evaluating collaboration
    # between your agents.
    print("\nExample end-of-episode info dump:\n", info)

#random_run()