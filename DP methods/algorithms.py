import numpy as np     
    
def PolicyEvaluation(env, policy, values, discount, state_space, action_space):

    for state in range(state_space):
        probability, nextstate, reward, _ = env.P[state][policy[state]][0]
        values[state] = reward + discount * probability * values[nextstate]

    return values

def PolicyImprovement(env, policy, values, discount, state_space, action_space):

    for state in range(state_space):

        temp = []
        for action in range(action_space):

            probability, nextstate, reward, _ = env.P[state][action][0]
            temp.append(reward + discount * probability * values[nextstate])
        policy[state] = np.argmax(temp)

    return policy

def ValueIteration(env, values, discount, state_space, action_space):

    for state in range(state_space):

        temp_values = []
        for action in range(action_space):

            probability, nextstate, reward, _ = env.P[state][action][0]
            temp_values.append(reward + discount * probability * values[nextstate])
        values[state] = np.max(temp_values)

    return values
