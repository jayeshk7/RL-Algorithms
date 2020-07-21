import numpy as np
import random as rd 

def generateEpisode(env):

    state = env.reset()  # env.reset() returns 3 things - current hand (total sum), dealer show card and usable ace (bool)
    episodes = []
    usable_ace = state[2]
    dealer_hand = state[1]
    curr_hand = state[0]
    while(1):

        action = 0 if curr_hand > 16 else 1     # policy which we evaluate, same as dealer's policy : 0 is stay 1 is hit
        nextstate, reward, done, _ = env.step(action)   # nextstate contains {player hand, dealer hand, usable ace}

        sample = (curr_hand, action, reward, usable_ace) 
        episodes.append(sample)
        if done:
            break
        curr_hand = nextstate[0]    # nextstate[0] is total sum of player hand
        usable_ace = nextstate[2]
    
    return episodes


def Policyimproving(env, policy):

    episodes = []
    state = env.reset()
    curr_hand = state[0]
    dealer_hand = state[1]
    usable_ace = state[2]
    
    while(1):

        action = policy[(curr_hand, dealer_hand, usable_ace)]
        nextstate, reward, done, _ = env.step(action)

        sample = (curr_hand, dealer_hand, action, reward, usable_ace)
        episodes.append(sample)
        if done:
            break
        curr_hand = nextstate[0]
        dealer_hand = nextstate[1]
        usable_ace = nextstate[2]
    
    return episodes


def Softmax(x):

    return np.exp(x) / np.sum(np.exp(x))