import numpy as np
import random as rd

class GaussianBandit(object):

    def __init__(self, num_arms, mean=10, variance=1):

        self.variance = variance
        self.mean = [rd.uniform(-mean, mean) for i in range(num_arms)]
        self.max_value_arm = np.argmax(self.mean)

    def Rewards(self, arm):

        return rd.gauss(self.mean[arm], self.variance)


def Softmax(arm_preference):

    a = np.exp(arm_preference)
    print(a)
    z = np.sum(np.exp(arm_preference))
    print(z)
    return a/z 


def GradientBandit(num_arms, alpha, iterations):

    arm_preference = np.zeros(num_arms)
    arm_reward = np.zeros(num_arms)
    arm_occur = np.zeros(num_arms)
    avg_reward = np.zeros(num_arms)

    for i in range(iterations):

        arm = int(rd.uniform(0,num_arms)) 
        arm_occur[arm] += 1

        reward = bandit.Rewards(arm)
        arm_reward[arm] += reward
        avg_reward[arm] = avg_reward[arm] + (reward - avg_reward[arm])/arm_occur[arm]

        softmax = Softmax(arm_preference)

        for a in range(num_arms):

            if(a == arm):
                arm_preference[a] = arm_preference[a] + alpha*(reward - avg_reward[a])*(1 - softmax[a])
            else:
                arm_preference[a] = arm_preference[a] + alpha*(reward - avg_reward[a])*softmax[a]

        print(arm_preference)

    print('here\n')
    if(i%1000 == 0):
        print('\nrewards are - ', arm_reward)
        print('\narm with max preference is - ', np.argmax(arm_preference))
        print('\narm with max expected return is - ', bandit.max_value_arm)
        print('-----------------------------------------------------------')


num_arms = 10
mean = 10
variance = 1
alpha = 0.5
iterations = 5000

bandit = GaussianBandit(num_arms, mean, variance)
print(bandit.mean)
print('----------------------------------------------------')

GradientBandit(num_arms, alpha, iterations)