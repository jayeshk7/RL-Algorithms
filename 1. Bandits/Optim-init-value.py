import numpy as np 
import random as rd

class GaussianBandit(object):

    def __init__(self, num_arms, variance=1, mean=10):
        
        self.mean = [rd.uniform(-mean,mean) for i in range(num_arms)]
        self.max_value_arm = np.argmax(self.mean)
        self.variance = variance

    def Rewards(self, arm):

        return rd.gauss(self.mean[arm], self.variance)


def OptimValue(optim_value, alpha, num_arms, iterations):

    q_value = optim_value*(np.ones(num_arms))
    arm_occur = np.zeros(num_arms)
    arm_reward = np.zeros(num_arms)

    arm = int(rd.uniform(0,num_arms))

    for i in range(iterations):
        
        arm_occur[arm] += 1
        reward = bandit.Rewards(arm)
        arm_reward[arm] += reward

        q_value[arm] = q_value[arm] + alpha*(reward - q_value[arm])

        arm = np.argmax(q_value)

        if(i%1000 == 0):
            print('\nrewards are - ', arm_reward)
            print('-----------------------------------------------------------')
    print('\narm chosen maximum number of times', np.argmax(arm_occur)+1)
    print('arm with maximum true expected reward', bandit.max_value_arm+1)

optim_value = 30
num_arms = 10
variance = 1
mean = 10
alpha = 0.5
iterations = 5000

bandit = GaussianBandit(num_arms, variance, mean)
print(bandit.mean)
print('\narm with max expected return is - ', bandit.max_value_arm + 1)
print('----------------------------------------------------')

OptimValue(optim_value, alpha, num_arms, iterations)
