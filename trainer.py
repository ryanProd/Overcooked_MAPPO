import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

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
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(env.observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
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

class Trainer(object):
    def __init__(self, env, device):
        self.lr = 2.5e-4
        self.epsilon = 1e-5
        self.num_steps = 400
        self.mini_batch_size = 100
        self.num_episodes = 1000
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.update_epochs = 4
        #clipped loss
        self.clip_coef = 0.2
        #entropy
        self.ent_coef = 0.01
        #value function
        self.vf_coef = 0.5
        #max norm for gradient clipping
        self.max_grad_norm = 0.5
        self.device = device
        self.agent = Agent(env).to(self.device)
        self.optimizer = optim.Adam(self.agent.parameters(), lr=self.lr, eps=self.epsilon)

        #buffer
        self.obs = torch.zeros((self.num_steps, ) + env.observation_space.shape).to(self.device)
        self.actions = torch.zeros((self.num_steps, ) + env.action_space.shape).to(self.device)
        self.logprobs = torch.zeros((self.num_steps, )).to(self.device)
        self.rewards = torch.zeros((self.num_steps, )).to(self.device)
        self.dones = torch.zeros((self.num_steps, )).to(self.device)
        self.values = torch.zeros((self.num_steps, )).to(self.device)
        self.advantages = torch.zeros_like(self.rewards).to(self.device)
        self.returns = torch.zeros_like(self.advantages).to(self.device)

    #learning rate annealing
    def lr_Anneal(self, eps, total_eps):
        frac = 1.0 - (eps - 1.0) / total_eps
        new_lr= frac * self.lr
        self.optimizer.param_groups[0]["lr"] = new_lr

    #needs to take in last done and last value
    def calculate_GAE(self, done, ob):
        with torch.no_grad():
            next_done = done
            next_value = self.agent.get_value(ob)
            lastgaelam = 0
            for t in reversed(range(self.num_steps)):
                if t == self.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                else:
                    nextnonterminal = 1.0 - self.dones[t+1]
                    next_value = self.values[t+1]
                delta = self.rewards[t] + self.gamma * next_value * nextnonterminal - self.values[t]
                self.advantages[t] = lastgaelam = delta + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam
            self.returns = self.advantages + self.values
        
    def train(self):
        b_inds = np.arange(self.num_steps)
        for epoch in range(self.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, self.num_steps, self.mini_batch_size):
                end = start + self.mini_batch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = self.agent.get_action_and_value_tensor(self.obs[mb_inds], self.actions.long()[mb_inds])
                logratio = newlogprob - self.logprobs[mb_inds]
                ratio = logratio.exp()

                mb_advantages = self.advantages[mb_inds]
                #normalize advantage
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                #clipped policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                #value loss
                v_loss = 0.5 * ((newvalue - self.returns[mb_inds]) ** 2).mean()
                entropy_loss = entropy.mean()

                loss = pg_loss - self.ent_coef * entropy_loss + v_loss * self.vf_coef

                #update weights
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm(self.agent.parameters(), self.max_grad_norm)
                self.optimizer.step()
    
    def save_state(self, path, global_steps):
            state = {
                'global_steps': global_steps,
                'state_dict': self.agent.state_dict(),
                'optimizer_dict': self.optimizer.state_dict(), 
            }
            torch.save(state, path)

    def add_ob(self, step, ob):
        self.obs[step] = ob
    def add_action(self, step, action):
        self.actions[step] = action
    def add_logprob(self, step, logprob):
        self.logprobs[step] = logprob
    def add_reward(self, step, reward):
        self.rewards[step] = reward
    def add_done(self, step, done):
        self.dones[step] = done
    def add_value(self, step, value):
        self.values[step] = value
    
