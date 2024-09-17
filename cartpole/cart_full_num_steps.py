import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

gym_id = "CartPole-v1"
lr = 2.5e-4
epsilon = 1e-5
num_steps = 400
mini_batch_size = 100
num_episodes = 1000
gamma = 0.99
gae_lambda = 0.95
update_epochs = 4
#clipped loss
clip_coef = 0.2
#entropy
ent_coef = 0.01
#value function
vf_coef = 0.5
#max norm for gradient clipping
max_grad_norm = 0.5
state_save_path = "cart_state.pt"
model_save_path = "cart_model.pt"
save_freq = 10

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    def __init__(self, env):
        super(Agent, self).__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(env.observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(env.observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, env.action_space.n), std=0.01),
        )

    def get_value(self, x):
        return self.critic(x).item()

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action.item(), probs.log_prob(action).item(), probs.entropy(), self.critic(x).item()
    
    def get_action_and_value_tensor(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)


def run():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = gym.make(gym_id)
    print(env.observation_space.shape)
    agent = Agent(env).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=lr, eps=epsilon)

    obs = torch.zeros((num_steps, ) + env.observation_space.shape).to(device)
    actions = torch.zeros((num_steps, ) + env.action_space.shape).to(device)
    logprobs = torch.zeros((num_steps, )).to(device)
    rewards = torch.zeros((num_steps, )).to(device)
    dones = torch.zeros((num_steps, )).to(device)
    values = torch.zeros((num_steps, )).to(device)

    global_steps = 0

    reward_per_done = []
    
    for eps in range(num_episodes):

        done = False

        ob = env.reset()
        ob = torch.from_numpy(ob[0]).to(device)

        done_reward = 0

        for step in range(num_steps):
            global_steps += 1

            obs[step] = ob
            

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(ob)
                values[step] = value
            actions[step] = action
            logprobs[step] = logprob


            ob, reward, done, truncated, info = env.step(action)
            rewards[step] = reward
            done_reward += reward

            
            dones[step] = done
            
            ob = torch.from_numpy(ob).to(device)
            if (done):
                #reset and keep going
                reward_per_done.append(done_reward)
                print(done_reward)
                done_reward = 0
                if (len(reward_per_done) > 31):
                    print("Mean Reward for Past 30 Dones:" + str(np.mean(reward_per_done[:-30])))
                ob = env.reset()
                ob = torch.from_numpy(ob[0]).to(device)
                done = False
        """
        print(obs)
        print(actions)
        print(logprobs)
        print(rewards)
        print(dones)
        print(values)
        """

        #rollout is done
        #Calculate GAE
        with torch.no_grad():
            #last done and value
            next_done = done
            next_value = agent.get_value(ob)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                else:
                    nextnonterminal = 1.0 - dones[t+1]
                    next_value = values[t+1]
                delta = rewards[t] + gamma * next_value * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values
        
        #sample mini batches
        b_inds = np.arange(num_steps)
        for epoch in range(update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, num_steps, mini_batch_size):
                end = start + mini_batch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value_tensor(obs[mb_inds], actions.long()[mb_inds])
                logratio = newlogprob - logprobs[mb_inds]
                ratio = logratio.exp()

                mb_advantages = advantages[mb_inds]
                #normalize advantage
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                #clipped policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                #value loss
                v_loss = 0.5 * ((newvalue - returns[mb_inds]) ** 2).mean()
                entropy_loss = entropy.mean()

                loss = pg_loss - ent_coef * entropy_loss + v_loss * vf_coef

                #update weights
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm(agent.parameters(), max_grad_norm)
                optimizer.step()
        
        #save model and state
        if (eps % save_freq == 0):
            state = {
                'global_steps': global_steps,
                'state_dict': agent.state_dict(),
                'optimizer_dict': optimizer.state_dict(), 
            }
            torch.save(state, state_save_path)


run()

