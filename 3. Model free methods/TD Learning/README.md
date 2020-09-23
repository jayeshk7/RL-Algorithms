# Temporal-Difference Learning

### Summary
Temporal difference learning is a family of algorithms in which the agent estimates the Q-values by bootstrapping from the current estimate of Q-values. Advantages of bootstrapping are reduced variance and faster propagation of reward signals.


**Problem Statement** : [Taxi environment](https://github.com/openai/gym/blob/master/gym/envs/toy_text/taxi.py). There are 4 locations (labeled by different letters) and the agent's job is to pick up the passenger at one location and drop him off in another. The agent receives +20 points for a successful dropoff, and loses 1 point for every timestep it takes. There is also a 10 point penalty for illegal pick-up and drop-off actions.

### Algorithms implemented:
- [x] SARSA
- [x] Q-Learning/SARSAMAX
- [ ] Double Q-Learning
- [ ] Expected SARSA

### Results
(results ready need to put gifs and graph)

### Resources:
1. Chapter 5 from [Sutton and Barto](http://incompleteideas.net/book/RLbook2020.pdf)
2. David Silver's course - [Lectures 4 and 5](https://www.youtube.com/playlist?list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ)
3. Stanford CS234 - [Lectures 3 and 4](https://www.youtube.com/playlist?list=PLoROMvodv4rOSOPzutgyCTapiGlY2Nd8u) 
