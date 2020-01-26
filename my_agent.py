from collections import deque
import torch
import torch.nn as nn
import numpy as np
import random

class my_actor_model ( nn.Module ):
    def __init__(self, in_sz, out_sz, hd_sz=0):
        super().__init__()
        self.l1 = nn.Linear(in_sz,hd_sz)
        self.l2 = nn.Linear(hd_sz,out_sz)
        self.leakyrelu = nn.LeakyReLU(0.1)

    def forward(self,states):
        x = states
        x = self.leakyrelu(self.l1(x))
        x = torch.tanh(self.l2(x))
        return x

class my_critic_model ( nn.Module ):
    def __init__(self, in_sz1, in_sz2, out_sz, hd_sz=0):
        super().__init__()
        self.l1 = nn.Linear(in_sz1, in_sz1*2)
        self.l2 = nn.Linear(in_sz2, in_sz2*2)
        self.l3 = nn.Linear((in_sz1+in_sz2)*2,hd_sz)
        self.l4 = nn.Linear(hd_sz,out_sz)
        self.leakyrelu = nn.LeakyReLU(0.1)

    def forward(self,states,actions):
        x = self.leakyrelu(self.l1(states))
        y = torch.tanh(self.l2(actions))
        z = torch.cat((x,y),dim=1)
        z = self.leakyrelu(self.l3(z))
        #x = torch.tanh(self.l2(x))
        z = self.leakyrelu(self.l4(z))
        return z


class replay_buffer:
    def __init__(self, buff_sz):
        self.buff = deque(maxlen=buff_sz)

    def push(self, state, action, next_state, reward, done):
        self.buff.append((state, action, next_state, reward, done))

    def sample(self, sz):
        states = []
        actions = []
        next_states = []
        rewards = []
        dones = []

        batch = random.sample(self.buff, sz)

        for i in batch:
            state, action, next_state, reward, done = i
            states.append(state)
            actions.append(action)
            next_states.append(next_state)
            rewards.append(reward)
            dones.append(done)

        return states, actions, next_states, rewards, dones

    def buff_len(self):
        return len(self.buff)

class my_ddpg_agent:
    def __init__(self,
            state_sz,
            action_sz,
            action_high,
            action_low,
            rp_buff_sz = 10000,
            train_epochs = 3,
            sample_sz = 100,
            actor_lr = 1e-4,
            critic_lr = 1e-4,
            actor_gamma = 0.99,
            critic_gamma = 0.99,
            learn_after_steps = 100,
            tau = 1e-3
            ):
        self.actor_m = my_actor_model(in_sz=state_sz, out_sz=action_sz, hd_sz=state_sz*action_sz)
        self.actor_target_m = my_actor_model(in_sz=state_sz, out_sz=action_sz, hd_sz=state_sz*action_sz)

        self.critic_m = my_critic_model(in_sz1=state_sz, in_sz2=action_sz, out_sz=1, hd_sz=state_sz*action_sz)
        self.critic_target_m = my_critic_model(in_sz1=state_sz, in_sz2=action_sz, out_sz=1, hd_sz=state_sz*action_sz)

        self.critic_loss_fn = nn.MSELoss()

        self.actor_optim_fn = torch.optim.Adam(self.actor_m.parameters(),lr=actor_lr)
        self.critic_optim_fn = torch.optim.Adam(self.critic_m.parameters(),lr=critic_lr)

        self.rp_buff_positive = replay_buffer(rp_buff_sz)
        self.rp_buff_negative = replay_buffer(rp_buff_sz)
        self.learn_step = 0
        self.action_step = 0
        self.learn_after_steps = learn_after_steps
        self.sample_sz = sample_sz
        self.train_epochs = train_epochs
        self.critic_gamma = critic_gamma
        self.actor_gamma = actor_gamma
        self.tau = tau
        self.noise = OUNoise(action_sz, action_high, action_low)


    def act(self,state,train_mode=True):
        self.action_step += 1
        self.actor_m.eval()
        with torch.no_grad():
            action = self.actor_m(torch.FloatTensor([state])).detach().squeeze()
        if(train_mode):
            action = self.noise.get_action(action.numpy(),self.learn_step)
        else:
            action = action.numpy()
        return action

    def reset(self):
        self.noise.reset()

    def step(self, state, action, reward, next_state, done):
        if(reward > 0):
            self.rp_buff_positive.push(state, action, [reward], next_state, [done])
        else:
            self.rp_buff_negative.push(state, action, [reward], next_state, [done])

        self.learn_step += 1

        if( self.learn_step % self.learn_after_steps != 0 ):
            return

        self.actor_m.train()
        self.critic_m.train()

        #print("ready to learn. learn_step {}, learn_after_steps {}, sample_sz {}".format(
        #    self.learn_step, self.learn_after_steps, self.sample_sz ))

        for i in range(self.train_epochs):
            psz = self.rp_buff_positive.buff_len()
            nsz = int(self.sample_sz/2)
            if(psz < nsz):
                nsz = self.sample_sz - psz
            elif(psz > nsz):
                psz = nsz

            (states_n, actions_n, rewards_n, next_states_n, dones_n) = self.rp_buff_negative.sample(nsz)
            (states_p, actions_p, rewards_p, next_states_p, dones_p) = self.rp_buff_positive.sample(psz)

            # Convert lists to Tensors
            states = torch.FloatTensor(states_n + states_p)
            actions = torch.FloatTensor(actions_n + actions_p)
            rewards = torch.FloatTensor(rewards_n + rewards_p)
            next_states = torch.FloatTensor(next_states_n + next_states_p)
            dones = torch.FloatTensor(dones_n + dones_p)

            #compute critic_m loss
            # y = r + gamma*q_target(next_state, target_next_action)
            target_next_actions = self.actor_target_m(next_states).detach()
            target_q_nexts = self.critic_target_m(next_states, target_next_actions)
            ys = rewards + (1-dones)*self.critic_gamma*target_q_nexts
            qs = self.critic_m(states,actions)
            critic_loss = self.critic_loss_fn(qs,ys)
            #print("target_q_nexts.shape: {}, ys.shape: {}, qs.shape: {}, rewards.shape{}, dones.shape{}".format(
            #    target_q_nexts.shape, ys.shape, qs.shape, rewards.shape, dones.shape ))

            #computer actor_m loss
            # actor doesnt have a loss. Actors goal is to maximize expected rewards
            # So, to teach actor, we maximize mean of Q(s,a)
            qs = self.critic_m(states, self.actor_m(states))
            actor_loss = -1*qs.mean()

            #train critic_m
            self.critic_optim_fn.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm(self.critic_m.parameters(), 1)
            self.critic_optim_fn.step()

            #train actor_m
            self.actor_optim_fn.zero_grad()
            actor_loss.backward()
            self.actor_optim_fn.step()

            #update critic_target_m and actor_target_m
            for target_param, param in ( zip ( self.critic_target_m.parameters(), self.critic_m.parameters() )):
                target_param.data.copy_(param.data*self.tau + target_param.data*(1.0 - self.tau))

            for target_param, param in ( zip ( self.actor_target_m.parameters(), self.actor_m.parameters() )):
                target_param.data.copy_(param.data*self.tau + target_param.data*(1.0 - self.tau))

        self.actor_m.eval()
        self.critic_m.eval()


"""
Taken from https://github.com/vitchyr/rlkit/blob/master/rlkit/exploration_strategies/ou_strategy.py
"""
class OUNoise(object):
    def __init__(self, action_space_dim, action_space_high, action_space_low, mu=0.0, theta=0.15, max_sigma=0.3, min_sigma=0.3, decay_period=100000):
        self.mu           = mu
        self.theta        = theta
        self.sigma        = max_sigma
        self.max_sigma    = max_sigma
        self.min_sigma    = min_sigma
        self.decay_period = decay_period
        self.action_dim   = action_space_dim
        self.low          = action_space_low
        self.high         = action_space_high
        self.reset()
        
    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu
        
    def evolve_state(self):
        x  = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state
    
    def get_action(self, action, t=0): 
        ou_state = self.evolve_state()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
        return np.clip(action + ou_state, self.low, self.high)
