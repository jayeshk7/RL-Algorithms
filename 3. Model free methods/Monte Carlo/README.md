# Monte Carlo solutions to Model-free RL

### Summary
Let's say we sampled a trajectory from our MDP. Now to judge how good the policy is we would look at the total return of the trajectory. Similarly, to judge how good the encountered state-action pairs were we could use the total return *starting from that state-action pair*. This is the basic idea behind Monte Carlo control where we sample trajectories and estimate our Q-values with total return we get starting from that state, and we improve our policy using our knowledge of our estimated Q-values.

**Problem Statement** : [Blackjack environment](https://github.com/openai/gym/blob/master/gym/envs/toy_text/blackjack.py)

### Algorithms implemented:
- [x] On-policy MC evaluation and control (Blackjack environment)
- [ ] Off-policy MC evaluation and control (Gridworld environment)

### Resources:
1. Chapter 5 from Sutton and Barto
2. David Silver's course - [Lectures 4 and 5]()
3. Stanford CS234 - Lectures 3 and 4

Feel free to raise an issue/PR if you find something wrong with the code!
