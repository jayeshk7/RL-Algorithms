import numpy as np
import gym
import time

env = gym.make('Taxi-v2')
env.unwrapped

print(env.observation_space)       # Discrete(500)
print(env.action_space)            # Discrete(6)
action_space = [0,1,2,3,4,5]

q_values = {}
policy = {}

for i in range(env.observation_space.n):                        
    policy.update({i : np.random.choice(action_space)})     # initialise policy
    for j in range(env.action_space.n):     
        q_values.update({(i,j) : 0})                        # initialise q_value table

epsilon = 0.9
alpha = 0.1
total_reward = 0
state = env.reset() 

for iterations in range(100000):

    nextstate, reward, done, _ = env.step(policy[state])
    total_reward += reward

    temp_q = []
    for action in range(env.action_space.n):
        temp_q.append(q_values[(nextstate, action)])
    target = reward + max(temp_q)

    # action value update
    q_values[(state, policy[state])] += alpha*(target - q_values[(state, policy[state])])
    state = nextstate

    # random behaviour policy (epsilon greedy)
    for i in range(env.observation_space.n):

        temp = []
        for j in range(env.action_space.n):
            temp.append(q_values[(i, j)])
        temp = np.asarray(temp)
        
        chance = np.random.uniform(0,1)
        if chance < epsilon:
            policy[i] = env.action_space.sample()
        else:
            policy[i] = np.argmax(temp)

    if done:
        state = env.reset()

    if (iterations+1)%10000 == 0:
        print(f'iterations completed : {iterations+1}, total reward = {total_reward}')
        epsilon /= 1.5


# forming the policy
for i in range(env.observation_space.n):
    temp = []
    for j in range(env.action_space.n):
        temp.append(q_values[(i,j)])
    policy[state] = np.argmax(temp)

# try out the policy
state = env.reset()
done = False
while not done:
    env.render()
    time.sleep(0.5)
    nextstate, reward, done, _ = env.step(policy[state])
    state = nextstate
