import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter


def generate_episode(env, policy):

    action_space = env.action_space.n
    experience = []
    rewards = []
    state = env.reset()
    action_probs = policy.forward(state)       
    action = int(np.random.choice(np.arange(action_space), p=action_probs.detach().numpy()))     # SELECTING ACTION ACCORDING TO THE PROBABILITY GIVEN BY NETWORK

    done = False
    while not done:
        nextstate, reward, done, _ = env.step(action)
        rewards.append(reward)                                   # APPENDING REWARD AT EACH STEP
        experience.append((state, action, reward, nextstate))    # APPENDING (S,A,R,S') TUPLE
        state = nextstate
        action_probs = policy.forward(state) 
        action = int(np.random.choice(np.arange(action_space), p=action_probs.detach().numpy()))     # SELECTING ACTION
    
    return experience, rewards

def plot(rewards):
    plt.plot(rewards)
    plt.xlabel('Episodes')
    plt.ylabel('Rewards')
    plt.plot(gaussian_filter(rewards, sigma=50))
    plt.show()