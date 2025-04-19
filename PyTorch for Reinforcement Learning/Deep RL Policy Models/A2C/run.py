import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium.vector import SyncVectorEnv, AsyncVectorEnv
import matplotlib.pyplot as plt
import argparse

class ActorPolicyNetwork(nn.Module):
    def __init__(self,
                 input_state_features=8, 
                 num_actions=4,
                 hidden_features=128):
        
        super(ActorPolicyNetwork, self).__init__()

        self.fc1 = nn.Linear(input_state_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, hidden_features)
        self.fc3 = nn.Linear(hidden_features, num_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        pi = F.softmax(self.fc3(x), dim=-1)
        return pi

class CriticValueNetwork(nn.Module):
    def __init__(self, input_state_features=8, hidden_features=128):
        super(CriticValueNetwork, self).__init__()
        self.fc1 = nn.Linear(input_state_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, hidden_features)
        self.fc3 = nn.Linear(hidden_features, 1)  

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        value = self.fc3(x)
        return value

def make_env():
    return gym.make("LunarLander-v3")

def train(num_envs, 
          n_steps=5,
          input_state_features=8, 
          num_actions=4, 
          hidden_features=128, 
          learning_rate=0.0005, 
          episodes=5000, 
          running_avg_steps=20, 
          print_freq=50, 
          gamma=0.99, 
          entropy_weight=0.005, 
          device="cpu"):

    if num_envs > 1:
        envs = AsyncVectorEnv([make_env for _ in range(num_envs)])
    else:
        envs = make_env()
        
    policy_network = ActorPolicyNetwork(
        input_state_features=input_state_features, 
        num_actions=num_actions,
        hidden_features=hidden_features
    ).to(device)
    
    value_network = CriticValueNetwork(
        input_state_features=input_state_features, 
        hidden_features=hidden_features
    ).to(device)

    p_optimizer = optim.Adam(policy_network.parameters(), lr=learning_rate)
    v_optimizer = optim.Adam(value_network.parameters(), lr=learning_rate)

    log = {"scores": [], 
           "running_avg_scores": []}
    for i in range(num_envs):
        log[f"env_{i}"] = []

    for episode in range(episodes):

        ### Get States for All Environments ###
        states, _ = envs.reset()
        states = torch.tensor(states, dtype=torch.float32, device=device)

        ### Compute Total Rewards Per Environment ###
        total_rewards = torch.zeros(num_envs)

        ### Done Flag Per Environment ###
        done = torch.zeros(num_envs, dtype=bool)

        ### Cache for Returns ###
        states_cache = []
        actions_probs_cache = []
        actions_cache = []
        rewards_cache = []
        log_probs_cache = []
        values_cache = []
        dones_cache = []

        
        ### Check if All Environments Are Completed ###
        while not torch.all(done):

            ### Compute Policy and Value ###
            action_probs = policy_network(states)
            values = value_network(states).squeeze(-1)

            ### Sample Action for Next Step (Flattened to a vector of length num_envs) ###
            actions = torch.multinomial(action_probs, num_samples=1)

            ### Compute the Log Probs of those Actions ###
            env_idx = torch.arange(num_envs)
            sampled_action_probs = torch.gather(action_probs, index=actions, dim=-1)
            log_probs = torch.log(sampled_action_probs).squeeze(-1)
            
            ### Step All The Environments ###
            actions_prepped = actions.squeeze().cpu().tolist()
            next_states, rewards, terminateds, truncateds, _ = envs.step(actions_prepped)

            ### Check Each Environment If They Are Done ###
            terminateds, truncateds = torch.tensor(terminateds, dtype=torch.bool), torch.tensor(truncateds, dtype=torch.bool) 
            done_flag = torch.logical_or(terminateds, truncateds)

            ### Convert Rewards/Next States to Tensor ###
            rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
            next_states = torch.tensor(next_states, dtype=torch.float32, device=device)
            
            ### Store Transitions ###
            states_cache.append(states)
            actions_probs_cache.append(action_probs)
            actions_cache.append(actions)
            rewards_cache.append(rewards)
            log_probs_cache.append(log_probs)
            values_cache.append(values)
            dones_cache.append(done_flag)

            ### Compute Total Rewards (for envs that are not done) ###
            total_rewards += (rewards * (~done.to(rewards.device))).cpu()
 
            ### Set Which Environments are Done ###
            done = torch.logical_or(done, done_flag)

            ### Set the State As the Next State ###
            states = next_states
            
            if len(rewards_cache) >= n_steps:

                ### Get the Value of the Next State (per environment) ###
                with torch.no_grad():
                    next_values = value_network(next_states).squeeze(-1)

                ### If the Environment is already done, the next state value should be 0 ###
                next_value = next_values * (~dones_cache[-1].to(next_values.device))

                ### Compute Returns ###
                returns = []
                G_t = next_values

                for t in reversed(range(n_steps)):

                    ### We have Multiple Environments that could be done ###
                    ### We need to make sure we dont change any reward or give ###
                    ### any reward to their done portions. Because while one game ###
                    ### may be done, other are still going! This is why we store the dones_cache ###
                    ### To just zero out any contribution until we get to the state where the env was ###
                    ### actually doing something! ###
                    G_t = rewards_cache[t] + gamma * G_t * (~dones_cache[t].to(rewards_cache[t].device))

                    ### Prepend Computed Returns ###
                    returns.insert(0, G_t)

                ### Convert stored Returns, log_probs, action_probs, and values to tensor ###
                returns = torch.stack(returns)
                log_probs = torch.stack(log_probs_cache)
                values = torch.stack(values_cache)
                action_probs = torch.stack(actions_probs_cache)

                ### Compute Advantage: A = G_t - V ###
                advantages = returns - values

                ### Compute Entropy to Avoid Overconfidence P * log(P) ###
                entropy = -torch.sum(action_probs * torch.log(action_probs + 1e-8))
                
                ### Compute Policy Gradient Loss (dont want value grads flowing to policy grads) and Entropy ###
                policy_loss = - torch.sum(log_probs * advantages.detach()) - entropy_weight * entropy

                ### Critic (Value) Loss ###
                value_loss = F.smooth_l1_loss(values, returns.detach())

                ### Sum Together All Losses ###
                total_loss = policy_loss + value_loss

                ### Zero Gradients and Update ###
                p_optimizer.zero_grad()
                v_optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(policy_network.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(value_network.parameters(), max_norm=1.0)
                p_optimizer.step()
                v_optimizer.step()
                
                ### Clear Cache for the next n steps ###
                states_cache = []
                actions_probs_cache = []
                actions_cache = []
                rewards_cache = []
                log_probs_cache = []
                values_cache = []
                dones_cache = []
                
        ### Logging ###
        log["scores"].append(torch.mean(total_rewards).item())
        log["running_avg_scores"].append(np.mean(log["scores"][-running_avg_steps:]))

        for n in range(num_envs):
            log[f"env_{n}"].append(total_rewards[n].item())

        if episode % print_freq == 0:
            print(f"Envs {num_envs}, Episode {episode}, Total Reward: {log["scores"][-1]:.2f}, Avg Reward: {log["running_avg_scores"][-1]:.2f}")

        if log["running_avg_scores"][-1] >= 200:
            print("Completed Training")
            break 
            
    return policy_network, value_network, log

if __name__ == "__main__":

    import pickle

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_envs", type=int)
    args = parser.parse_args()
    
    ### Play Game ###
    env = gym.make("LunarLander-v3")

    policy_network, value_network, log = train(args.num_envs, device="cuda")

    with open(f'results/envs_{args.num_envs}.pkl', 'wb') as handle:
        pickle.dump(log, handle, protocol=pickle.HIGHEST_PROTOCOL)
        

