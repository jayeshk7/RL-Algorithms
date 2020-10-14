# Deep Deterministic Policy Gradients

### Summary
This paper builds on the idea of [Deterministic policy gradients]() and uses function approximation to solve continuous control problems. This implementation tries to solve the inverted pendulum task.

### Results and plots

**Rewards v/s Episodes**
![rewardplot](https://github.com/jayeshk7/Deep-RL/blob/master/3.%20Deterministic%20PG/DDPG/Inverted%20pendulum.png)

**Observations** : The policy starts performing badly after a point and it keeps getting worse. This was observed in every run of this algorithm and I am not exactly sure why this happens.