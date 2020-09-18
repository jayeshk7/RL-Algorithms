# Model-free RL

### Summary
One drawback of dynamic programming methods is that it needs the transition dynamics of the environment to find the optimal policy. There is a different class of algorithms which can solve for the optimal policy without the knowledge of the MDP. The idea here is to sample trajectories from the MDP and estimate Q-values for all the encountered state-action pairs.

### Algorithms implemented :
- [x] On-policy Monte Carlo control
- [ ] Off-policy Monte Carlo control using importance sampling
- [x] Q-Learning
- [x] SARSA

### Resources
1. Chapter 4, 5 and 6 from [Sutton and Barto]()
2. David Silver's course - Lectures  4 and 5
3. Stanford CS234 - Lectures 3 and 4
