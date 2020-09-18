# Dynamic Programming Methods

### Summary
In settings where we know how the world/environment works, there are some simpler planning algorithms which we can use like Policy Iteration (PI) and Value Iteration (VI). Essentially, it's about trying to learn the true expected reward of each state and then making decisions using this knowledge of learnt state/action values. 

**Problem Statement** : [Frozen Lake environment](https://github.com/openai/gym/blob/master/gym/envs/toy_text/frozen_lake.py). The agent controls the movement of a character in a grid world. Some tiles of the grid are walkable, and others lead to the agent falling into the water. The agent is rewarded for finding a walkable path to a goal tile. 

### Algorithms implemented :
- [x] Policy Iteration
- [x] Value Iteration

### Results
(i dont have results rn just put a gif and reward graph)

### Resources
1. Chapters 2 and 3 from [Sutton and Barto]()
2. David Silver's course - lectures 1, 2 and 3
3. Stanford CS234 - lecture 1 and 2