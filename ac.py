import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import gymnasium as gym 
from functools import reduce 
from catch import Catch

# class ActorCritic(nn.Module):
#     def __init__(self, num_inputs, num_actions, hidden_size):
#         super(ActorCritic, self).__init__()
#         self.num_actions = num_actions
#         self.critic = nn.Sequential(
#             nn.Linear(num_inputs, hidden_size),
#             nn.ReLU(),
#             nn.Linear(hidden_size, 1)
#         )
#         self.actor = nn.Sequential(
#             nn.Linear(num_inputs, hidden_size),
#             nn.ReLU(),
#             nn.Linear(hidden_size, num_actions)
#         )

#     def forward(self, state):
#         print(state.flatten)
#         value = self.critic(state)    
        # actor_logits = self.actor(state)
        # actor_logits = torch.clamp(actor_logits, -10, 10) # clip logits to avoid negative probabilities
#         policy_dist = F.softmax(self.actor(state), dim=-1)
#         return value, policy_dist

class ActorCritic(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        
    
    def forward(self, x):
        x = torch.flatten(x)
        x = torch.relu(self.fc1(x))
        x_p=  torch.clamp(x, -10, 10) 
        policy = nn.functional.softmax(self.fc2(x_p), dim=-1)
        value = self.fc3(x)
        return policy, value
 

def a_c(env,input_dim, out_dim, hidden_size=128, num_episodes=1000, gamma=0.96, lr=1e-3):
    # env = env_fn()
    
    # actor_critic = ActorCritic(env.observation_space.shape[0], env.action_space.n, hidden_size)
    actor_critic = ActorCritic(input_dim, out_dim, hidden_size)
    optimizer = optim.Adam(actor_critic.parameters(), lr=lr)

    all_rewards = []
    for i in range(num_episodes):
        state = env.reset()
        log_probs = []
        values = []
        rewards = []
        # state_np = state[0]
        while True:
            value, policy_dist = actor_critic(torch.from_numpy(state).float())
            # print(policy_dist)
            # print(torch.abs(policy_dist))
            policy_dist = torch.abs(policy_dist)
            action = policy_dist.multinomial(num_samples=1).detach().item()
            # print(action)
            log_prob = F.log_softmax(policy_dist, dim=-1)[action]
            next_state, reward, done,* _ = env.step(action)
            log_probs.append(log_prob)
            values.append(value)
            rewards.append(reward)
            if done:
                all_rewards.append(sum(rewards))
                break
            state = next_state
        
        # Compute returns and advantages
        returns = []
        gae = 0
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + gamma * (0 if i == len(rewards) - 1 else values[i + 1].detach()) - values[i].detach()
            gae = delta + gamma * 0.97 * (0 if i == len(rewards) - 1 else gae)
            returns.insert(0, gae + values[i].detach())
        # print(returns)
        # if len(returns) == 1:
        #     returns = torch.FloatTensor(returns)
        # else:
        #     returns = torch.FloatTensor(returns[0])
        returns = torch.FloatTensor(returns[0])
        log_probs = torch.stack(log_probs)
        values = torch.stack(values)
        # print('values',values)
        # print(torch.tensor(returns))

        # Compute advantages with baseline subtraction
        advantages = returns - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Compute actor and critic losses
        actor_loss = -(log_probs * advantages.detach()).mean()
        critic_loss = F.smooth_l1_loss(values, returns.detach())

        # Backpropagate and update weights
        optimizer.zero_grad()
        (actor_loss + critic_loss).backward()
        optimizer.step()

        # Print progress
        if i % 10 == 0:
            print('Episode %d \t Average Reward: {:.2f}'.format(i, np.mean(all_rewards[-10:])))

    env.close()



rows = 7
columns = 7
speed = 1.0
max_steps = 250
max_misses = 10
observation_type = 'pixel' # 'vector'
seed = None
    
    # Initialize environment and Q-array
env = Catch(rows=rows, columns=columns, speed=speed, max_steps=max_steps,
                max_misses=max_misses, observation_type=observation_type, seed=seed)

state_space = env.observation_space.shape
action_space = env.action_space.n

# print(state_space)
# env = gym.make('CartPole-v1')

in_dim =  reduce(lambda x, y: x * y,state_space)
hidden_size=128
out_dim = action_space
a_c(env, in_dim,out_dim, hidden_size,num_episodes=1000, gamma=0.96, lr=0.001)