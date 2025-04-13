# PyTorch for Reinforcement Learning

Reinforcement Learning is one of the coolest types of learning methods out there, where instead of learning from input/output pairs like we typically do in supervised learning, we instead learn from experience. This repository is called PyTorch-Adventures, but I can't really move onto Deep Reinforcement Learning without setting the stage with more traditional RL methods. So we will break this section into a few parts:

## Iterative Reinforcement Learning

<img src="https://github.com/priyammaz/PyTorch-Adventures/blob/main/src/visuals/frozen_lake.gif?raw=true" alt="drawing" width="300"/>

In this section, we'll explore traditional reinforcement learning methods for solving simple environments. This will help introduce key RL concepts and provide a deeper understanding (and derivation) of the Bellman Equation. We'll cover two approaches: Model-Based Learning, where environment dynamics are known via a Markov Decision Process (MDP), and Model-Free Learning, where such information is unavailable.

**Model-Based Learning**
- [Policy Iteration](Intro%20to%20Reinforcement%20Learning/Model-Based%20Learning/intro_rl_and_policy_iter.ipynb)
- [Value Iteration](Intro%20to%20Reinforcement%20Learning/Model-Based%20Learning/value_iteration.ipynb)

**Model-Free Learning**
- [Monte-Carlo](Intro%20to%20Reinforcement%20Learning/Model-Free%20Learning/monte_carlo.ipynb)
- [SARSA](Intro%20to%20Reinforcement%20Learning/Model-Free%20Learning/sarsa.ipynb)
- [Q-Learning](Intro%20to%20Reinforcement%20Learning/Model-Free%20Learning/q_learning.ipynb)
  
## Deep Reinforcement Learning

<img src="https://github.com/priyammaz/PyTorch-Adventures/blob/main/src/visuals/lunarlander.gif?raw=true" alt="drawing" width="400"/>

Now that we've explored iterative methods for solving environments, we turn to Neural Networks to do the same at a larger scale. These approaches generally fall into three categories:

- **Value-based**: where the network estimates state-action values to derive a policy,
- **Policy-based**: where it directly learns the policy, and
- **Actor-Critic**: a hybrid of both.

Unlike traditional methods, neural networks can also handle continuous state and action spaces, expanding the range of problems we can solve.

**Deep Value Estimation**
- [Deep-Q Learning](Intro%20to%20Deep%20Reinforcement%20Learning/Deep%20Q-Learning/deep_q_learning.ipynb)
- [Double Deep-Q Learning](Intro%20to%20Deep%20Reinforcement%20Learning/Double%20Deep-Q%20Learning/double_deep_q_learning.ipynb)
- [Dueling Deep-Q Learning](Intro%20to%20Deep%20Reinforcement%20Learning/Dueling%20Deep-Q%20Learning/dueling_deep_q_learning.ipynb)
- [Prioritized Experience Replay](Intro%20to%20Deep%20Reinforcement%20Learning/Prioritized%20Experience%20Replay/prioritized_experience_replay.ipynb)
- [PER with SumTree](Intro%20to%20Deep%20Reinforcement%20Learning/Prioritized%20Experience%20Replay/sumtree_per.ipynb)


