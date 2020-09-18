# Multi-Armed Bandit 

### Summary
This is a classic reinforcement learning problem that exemplies the exploration-exploitation tradeoff. Each arm can be thought of as the arm of one slot machine. The rewards may differ every time we pull the arm but there's a certain expected reward associated with every slot machine. 
To find the expectated reward of each slot machine we would have to choose it again and again to see what kind of rewards we get, even if we don't get high immediate rewards. 
The agent needs to *explore* as well as *exploit* it's current knowledge of which arm yields high immediate reward. This aim of maximizing the objective of expected rewards leads to the exploration-exploitation dilemma. 

**Problem Statement** : There are 10 one-arm bandits. Reward associated with each arm is a Gaussian distribution with some fixed mean and variance. The mean of the gaussian represents the expected reward and is sampled randomly for each arm from some uniform distribution.

### Algorithms implemented
- [x] Epsilon Greedy
- [x] Softmax exploration
- [x] Optimistic init 
- [ ] UCB
- [ ] Median Elimination 
- [ ] Thompson Sampling

### Results
(i dont have results, generate results) 

### Resources
1. Chapters 1 and 2 from Introduction to Reinforcement Learning by Sutton and Barto