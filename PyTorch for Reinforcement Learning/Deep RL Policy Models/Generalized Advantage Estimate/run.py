"""
Copy of code from notebook, just wanted to test different lambdas!
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import gymnasium as gym
import matplotlib.pyplot as plt
import argparse

def compute_gae(rewards, values, gamma=0.99, lambda_weight=0.95):

    """
    This is the implemenation of the recursive form of our GAE! All we need to do is
    compute our return for the current step and then add it to the scaled advantage of
    the next step!
    """

    ### Store our Advantages and Returns ###
    advantages = []
    returns = []

    ### Our Initial Advantage and the next value (after the terminal state in our list of values) is 0 ###
    advantage = 0
    next_value = 0

    ### Loop through in Reverse ###
    for r, v in zip(reversed(rewards), reversed(values)):

        ### Compute Delta (our TD Error): delta = r + gamma * V(s_{t+1}) - V(s_t) ###
        td_error = r + next_value * gamma - v

        ### Compute our Advantage: A_t = delta + lambda * gamma * A_{t+1} ###
        advantage = td_error + advantage * gamma * lambda_weight
        
        ### Compute Return using relation: g_t = A + V ###
        g_t = advantage + v

        ### Set the Next Value as the current V ###
        next_value = v

        ### Prepend our Advantages and Return ###
        advantages.insert(0, advantage)
        returns.insert(0, g_t)

    ### Convert to tensors, and confirm dtype and device ###
    advantages = torch.tensor(advantages, dtype=torch.float32, device=values.device)
    returns = torch.tensor(returns, dtype=torch.float32, device=values.device)
    
    return advantages, returns

class PolicyNetwork(nn.Module):
    def __init__(self,
                 input_state_features=8, 
                 num_actions=4,
                 hidden_features=128):
        
        super(PolicyNetwork, self).__init__()

        self.fc1 = nn.Linear(input_state_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, hidden_features)
        self.fc3 = nn.Linear(hidden_features, num_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        pi = F.softmax(self.fc3(x), dim=-1)
        return pi

class ValueNetwork(nn.Module):
    def __init__(self, input_state_features=8, hidden_features=128):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(input_state_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, hidden_features)
        self.fc3 = nn.Linear(hidden_features, 1)  

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        value = self.fc3(x)
        return value


def train(env, 
          input_state_features=8, 
          num_actions=4,
          hidden_features=128,
          learning_rate=0.0003,
          episodes=3000, 
          running_avg_steps=10, 
          print_freq=50,
          gamma=0.99,
          lambda_weight=0.99,
          entropy_weight=0.01,
          device="cpu"):

    policy_network = PolicyNetwork(
        input_state_features=input_state_features, 
        num_actions=num_actions,
        hidden_features=hidden_features
    ).to(device)
    
    p_optimizer = optim.Adam(policy_network.parameters(), lr=learning_rate)

    value_network = ValueNetwork(
        input_state_features=input_state_features, 
        hidden_features=hidden_features
    ).to(device)

    v_optimizer = optim.Adam(value_network.parameters(), lr=learning_rate)

    log = {"scores": [], 
           "running_avg_scores": []}
    
    for i in range(episodes):

        state, _ = env.reset()
        log_probs = []
        rewards = []
        entropies = []
        done = False
        
        values = []
        
        while not done:

            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            
            action_probs = policy_network(state)
            
            selected_action = torch.multinomial(action_probs, num_samples=1).item()
            
            log_prob = torch.log(action_probs[0, selected_action])

            entropy = -torch.sum(action_probs * torch.log(action_probs + 1e-8))

            value = value_network(state)

            values.append(value)
            
            next_state, reward, terminal, truncated, _ = env.step(selected_action)
            
            log_probs.append(log_prob)
            rewards.append(reward)
            entropies.append(entropy)
            
            done = terminal or truncated
            
            state = next_state

        values = torch.cat(values, dim=0).squeeze()
        
        ##########################
        ### COMPUTE ADVANTAGES/RETURNS ###
        advantages, returns = compute_gae(rewards, values, gamma=gamma, lambda_weight=lambda_weight)

        ### NORMALIZE ADVANTAGES ###
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        ###########################
        
        policy_loss = -torch.sum(torch.stack(log_probs) * advantages) - entropy_weight * torch.sum(torch.stack(entropies))

        value_loss = F.smooth_l1_loss(values, returns)

        p_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(policy_network.parameters(), max_norm=1.0)
        p_optimizer.step()

        v_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(value_network.parameters(), max_norm=1.0)
        v_optimizer.step()

        total_rewards = sum(rewards)

        ### Log ###
        log["scores"].append(total_rewards)
        running_avg_score = np.mean(log["scores"][-running_avg_steps:])
        log["running_avg_scores"].append(running_avg_score)

        if i % print_freq == 0:
            print(f"Episode {i}, Total Reward: {total_rewards}, Average Reward: {running_avg_score}")

    return policy_network, value_network, log

if __name__ == "__main__":

    import pickle

    parser = argparse.ArgumentParser()
    parser.add_argument("--lambda_weight", type=float)
    args = parser.parse_args()
    
    ### Play Game ###
    env = gym.make("LunarLander-v3")

    policy_network, value_network, log = train(env, device="cuda", lambda_weight=args.lambda_weight)

    with open(f'results/lambda_{args.lambda_weight}.pkl', 'wb') as handle:
        pickle.dump(log, handle, protocol=pickle.HIGHEST_PROTOCOL)

