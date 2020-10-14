# Policy gradient methods

### Summary
Policy gradient methods is a class of algorithms where we directly optimize our policy. The policy is represented by a function approximator like neural networks which takes in the state and outputs a distribution over actions. Agent takes the action with the highest probability.<br/>
Since the policy is already stochastic in this case, we don't have to think about exploration here.

### Results

**REINFORCE on CartPole-v0**
![reinforce_cartpole](https://github.com/jayeshk7/Deep-RL/blob/master/2.%20Vanilla%20Policy%20gradient%20methods/REINFORCE/Cartpole_result.png)

