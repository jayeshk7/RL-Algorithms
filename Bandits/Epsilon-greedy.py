import numpy as np
import random

class GaussianBandit(object):

    def __init__(self, num_arms, variance=1, mean=10):
        
        self.mean = [random.uniform(-mean, mean) for i in range(num_arms)]
        self.max_value_arm = np.argmax(self.mean)
        self.variance = variance
    
    def Rewards(self, arm):

        return random.gauss(self.mean[arm], self.variance)


def EpsilonGreedy(epsilon, num_arms, iter, alpha):

    q_value = np.zeros(num_arms)
    arm_occur = np.zeros(num_arms)
    arm_reward = np.zeros(num_arms)

    for i in range(iter):

        rand = random.uniform(0,1)
        
        if(rand > epsilon):
            arm = np.argmax(q_value)
        else: 
            arm = int(random.uniform(0,num_arms))

        arm_occur[arm] += 1
        reward = bandit.Rewards(arm)
        arm_reward[arm] += reward

        q_value[arm] = q_value[arm] + alpha*(reward - q_value[arm])


        if(i%1000 == 0):
            epsilon /= 2
            print('\nrewards are - ', arm_reward)
            print('-----------------------------------------------------------')
    print('\narm chosen maximum number of times -', np.argmax(arm_occur)+1)
    print('arm with max expected return is - ', bandit.max_value_arm+1)


iter = 10000
num_arms = 10
alpha = 0.5
epsilon = 0.7
mean = 10
variance = 1

bandit = GaussianBandit(num_arms)
print('true expected rewards\n\n',bandit.mean,'\n')
print('---------------------------------------------------')

EpsilonGreedy(epsilon, num_arms, iter, alpha)
