import numpy as np
import random as rd 

def generateEpisode(env):

    state = env.reset()  # env.reset() returns 3 things - current hand, dealer show card and usable ace (bool)
    episodes = []
    dealer_hand = state[1]
    curr_hand = state[0]
    while(1):

        action = 0 if curr_hand > 16 else 1  # same as dealer's policy : 0 is stay 1 is hit
        nextstate, reward, done, _ = env.step(action)   # nextstate returns {player hand, dealer hand, usable ace}

        sample = (curr_hand, action, reward)
        episodes.append(sample)
        if done:
            break
        curr_hand = nextstate[0]    # nextstate[0] is total sum of player hand
    
    return episodes


def Policyimproving(env, policy):

    episodes = []
    state = env.reset()
    curr_hand = state[0]
    while(1):

        nextstate, reward, done, _ = env.step(policy[curr_hand])

        sample = (curr_hand, policy[curr_hand], reward)
        episodes.append(sample)
        if done:
            break
        curr_hand = nextstate[0]
    return episodes


def Softmax(x):

    return np.exp(x) / np.sum(np.exp(x))