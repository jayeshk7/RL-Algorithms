# Deterministic Policy Gradients

### Summary 
This family of deep RL algorithms build on top of the idea presented in [Deterministic Policy Gradients](). These assume the policy to be a deterministic function of the features and not a probability distribution over all actions. Deep deterministic policy gradients (DDPG) and Twin Delayed Deep Deterministic Policy Gradients (TD3) are some of the widely used algorithms.

These algorithms can be used to solve any continuous control task while the result may or may not be good depending on the difficulty of the task. TD3 usually outperforms DDPG and makes progress in environments even where DDPG fails.