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
import json


from trainer import Trainer
from video_recorder import VideoRecorder

## Swap between the 5 layouts here:
#solved
#layout = "cramped_room"
#solved
#layout = "asymmetric_advantages"
#stuck at 5
#layout = "coordination_ring"
#stuck at 6.9
#layout = "forced_coordination"
#stuck at 5.8
layout = "counter_circuit_o_1order"

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
#reward_shaping_horizon = 5e6
#for 4, 5
reward_shaping_horizon = 4e6

# Length of Episodes.  Do not modify for your submission!
# Modification will result in a grading penalty!
horizon = 400

#used to evaluate trained agents
def eval():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_episodes = 500
    num_steps = 400

    mdp = OvercookedGridworld.from_layout_name(layout, rew_shaping_params=reward_shaping)
    base_env = OvercookedEnv.from_mdp(mdp, horizon=horizon, info_level=0)
    env = gym.make("Overcooked-v0", base_env=base_env, featurize_fn=base_env.featurize_state_mdp)

    episode_rewards = []
    episode_soups = []

    cook_zero = Trainer(env, device)
    cook_one = Trainer(env, device)

    zero_state = torch.load("saved_models/" + layout + "_3_hidden_layers_zero.pt")
    one_state = torch.load("saved_models/" + layout + "_3_hidden_layers_zero.pt")

    cook_zero.agent.load_state_dict(zero_state['state_dict'])
    cook_zero.optimizer.load_state_dict(zero_state['optimizer_dict'])
    cook_one.agent.load_state_dict(one_state['state_dict'])
    cook_one.optimizer.load_state_dict(one_state['optimizer_dict'])
    global_steps = zero_state['global_steps']

    cook_zero.agent.eval()
    cook_one.agent.eval()

    other_metrics = [[], []]

    for eps in range(num_episodes):
        done = False
        obs = env.reset()
        reward_per_ep = 0
        soups_per_ep = 0
        for step in range(num_steps):
            ob_zero = torch.from_numpy(obs["both_agent_obs"][0]).type(torch.float32).to(device)
            ob_one = torch.from_numpy(obs["both_agent_obs"][1]).type(torch.float32).to(device)
        
            with torch.no_grad():
                action_zero, logprob_zero, _, value_zero = cook_zero.agent.get_action_and_value(ob_zero)
                action_one, logprob_one, _, value_one = cook_zero.agent.get_action_and_value(ob_one)
            
            obs, R, done, info = env.step([action_zero, action_one])

            soups_made = int(R / 20)
            if (soups_made > 0):
                print("SOOOOOOUUUUUUUP!")
                soups_per_ep += soups_made

            if (done):
                episode_rewards.append(reward_per_ep)
                episode_soups.append(soups_per_ep)
                
                other_metrics[0].append(len(info['episode']['ep_game_stats']["dish_drop"][0]) + len(info['episode']['ep_game_stats']["dish_drop"][1]))
                other_metrics[1].append(len(info['episode']['ep_game_stats']["soup_drop"][0]) + len(info['episode']['ep_game_stats']["soup_drop"][1]))
                print("Episode Reward: " + str(reward_per_ep))
                print("Episode Soups: " + str(soups_per_ep))
                print("--------------------")
                break
    
    with open('eval_data/' + layout + "other_metrics.json", 'w') as file:
        json.dump(other_metrics, file)


#used to run training
def run():
    num_episodes = 500000
    num_steps = 400
    save_frequency = 200

    mdp = OvercookedGridworld.from_layout_name(layout, rew_shaping_params=reward_shaping)
    base_env = OvercookedEnv.from_mdp(mdp, horizon=horizon, info_level=0)
    env = gym.make("Overcooked-v0", base_env=base_env, featurize_fn=base_env.featurize_state_mdp)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cook_zero = Trainer(env, device)
    cook_one = Trainer(env, device)

    global_steps = 0
    episode_rewards = []
    episode_soups = []

    #comment out everyhing underneath but above "for eps in range()" to start new training
    zero_state = torch.load("saved_models/" + layout + "_3_hidden_layers_zero.pt")
    one_state = torch.load("saved_models/" + layout + "_3_hidden_layers_zero.pt")

    cook_zero.agent.load_state_dict(zero_state['state_dict'])
    cook_zero.optimizer.load_state_dict(zero_state['optimizer_dict'])
    cook_one.agent.load_state_dict(one_state['state_dict'])
    cook_one.optimizer.load_state_dict(one_state['optimizer_dict'])
    global_steps = zero_state['global_steps']
    with open('training_data/' + layout + "_3_hidden_layers.json", 'r') as openfile:
        episode_soups = json.load(openfile)
    
    
    #recorder = VideoRecorder("videos/cramped_room")

    for eps in range(num_episodes):
        done = False
        obs = env.reset()
        num_soups_made = 0
        reward_per_ep = 0
        soups_per_ep = 0

        #lr anneal
        #cook_zero.lr_Anneal(eps, num_episodes)
        #cook_one.lr_Anneal(eps, num_episodes)
        
        #rollout
        for step in range(num_steps):
            global_steps += 1

            #testing this out
            ob_zero = torch.from_numpy(obs["both_agent_obs"][0]).type(torch.float32).to(device)
            ob_one = torch.from_numpy(obs["both_agent_obs"][1]).type(torch.float32).to(device)
            

            cook_zero.add_ob(step, ob_zero)
            cook_one.add_ob(step, ob_one)

            with torch.no_grad():
                action_zero, logprob_zero, _, value_zero = cook_zero.agent.get_action_and_value(ob_zero)
                action_one, logprob_one, _, value_one = cook_zero.agent.get_action_and_value(ob_one)
                cook_zero.add_value(step, value_zero)
                cook_one.add_value(step, value_one)

            cook_zero.add_action(step, action_zero)
            cook_one.add_action(step, action_one)

            cook_zero.add_logprob(step, logprob_zero)
            cook_one.add_logprob(step, logprob_one)

            obs, R, done, info = env.step([action_zero, action_one])

            #recorder.record(env)

            soups_made = int(R / 20)
            if (soups_made > 0):
                print("SOOOOOOUUUUUUUP!")
                soups_per_ep += soups_made
            num_soups_made += soups_made # Each served soup generates 20 reward
            r_shaped = info["shaped_r_by_agent"]
            #this is the default
            #shared_reward = R + ((r_shaped[0] + r_shaped[1]) * (1 - (global_steps / reward_shaping_horizon)))
            shared_reward = R + r_shaped[0] + r_shaped[1]
            reward_per_ep += shared_reward

            cook_zero.add_reward(step, shared_reward)
            cook_one.add_reward(step, shared_reward)
            cook_zero.add_done(step, done)
            cook_one.add_done(step, done)

            if (done):
                episode_rewards.append(reward_per_ep)
                episode_soups.append(soups_per_ep)
                print("Episode Reward: " + str(reward_per_ep))
                print("Episode Soups: " + str(soups_per_ep))
                break
        
        #rollout done
        print("Epsiode: " + str(eps))
        if (len(episode_rewards) > 31):
            print("Mean Reward Last 30 Episodes: " + str(np.mean(get_last_n(30, episode_rewards))))
        if (len(episode_soups) > 31):
            print("Mean Soups Made Last 30 Episodes: " + str(np.mean(get_last_n(30, episode_soups))))
            print(get_last_n(30, episode_soups))
        print("---------------------------------------")

        #recorder.save(str(eps)+".mp4")
        
        cook_zero.calculate_GAE(done, torch.from_numpy(obs["both_agent_obs"][0]).type(torch.float32).to(device))
        cook_one.calculate_GAE(done, torch.from_numpy(obs["both_agent_obs"][1]).type(torch.float32).to(device))

        cook_zero.train()
        cook_one.train()

        if eps % save_frequency == 0:
            cook_zero.save_state("saved_models/" + layout + "_3_hidden_layers_zero.pt", global_steps)
            cook_one.save_state("saved_models/" + layout + "_3_hidden_layers_one.pt", global_steps)
            with open('training_data/' + layout + "_3_hidden_layers.json", 'w') as file:
                json.dump(episode_soups, file)

def get_last_n(n, x):
    j = 0
    i = len(x) - 1
    out = np.zeros(n)
    y = n - 1
    while True:
        if (j >= n):
            break
        out[y] = x[i]
        i -= 1
        y -= 1
        j += 1
    return out

eval()
#run()