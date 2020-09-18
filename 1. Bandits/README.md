# Multi-Armed Bandit 

### Summary
This is a classic reinforcement learning problem that exemplies the exploration-exploitation tradeoff. Each arm can be thought of one arm of a slot machine. The rewards may differ every time we pull the arm but there's a true expected reward associated with every slot machine.<br/>
To find the true expectated reward we would have to choose one arm again and again to see what kind of rewards we get, even if we don't get high immediate rewards.<br/> 
The agent needs to *explore* as well as *exploit* it's current knowledge of which arm yields high immediate reward. This aim of maximizing the objective of expected rewards leads to the exploration-exploitation dilemma. 

**Problem Statement** : There are 10 one-arm bandits. Reward associated with each arm is a Gaussian distribution with some fixed mean and variance. The mean of the gaussian represents the expected reward and is sampled randomly for each arm from some uniform distribution.

### Algorithms implemented
- [x] Epsilon Greedy
- [x] Softmax exploration
- [x] Optimistic initialisation 
- [ ] UCB
- [ ] Median Elimination 
- [ ] Thompson Sampling

### Results
(no results right now, get some graphs/comparisons) 

### Resources
1. Chapters 1 and 2 from [Sutton and Barto](http://incompleteideas.net/book/RLbook2020.pdf)