import random
from collections import defaultdict

import numpy as np


class SARSA():
    """State–action–reward–state–action (SARSA) is an algorithm 
    for learning a Markov decision process policy, used in the 
    reinforcement learning area of machine learning.

    Parameters:
    -----------
    actions : list
        actions.
    learning_rate : float
        The learning rate or step size determines to what extent newly acquired information
        overrides old information. 
        0 = makes the agent learn nothing.
        1 = makes the agent consider only the most recent information
    discount factor : float
        A factor of 0 makes the agent "opportunistic" by only considering current rewards, 
        while a factor approaching 1 will make it strive for a long-term high reward.
    epsilon : float
         epsilon greedy strategy
    """

    def __init__(self, actions, learning_rate=0.01, discount_factor=0.9, epsilon=0.1):
        self.actions = actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon

        """Step 1: Initialize Q-values
        Q-Learning table of states by actions that is initialized to zero, 
        then each cell is updated through training.
        
        q_table = [state][action]
        state = [col, row]
        action = [up, down, left, right]

        {
            '[0, 0]': [0.0, 0.0, 0.0, 0.0], 
            '[0, 1]': [0.0, 0.0, 0.0, 0.0], 
            '[1, 1]': [0.0, -1.99, 0.0, -1.0], 
        }
        """
        self.q_table = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])

    def choose_action(self, state):
        """Step 2: Choose an action
        use the epsilon greedy strategy
        if random number < epsilon: random action
        if random number > epsilon: according to q table
        """
        if np.random.rand() < self.epsilon:
            action = np.random.choice(self.actions)
        else:
            state_action = self.q_table[state]
            action = self.arg_max(state_action)
        return action

    def arg_max(self, state_action):
        return random.choice([index for index, value in enumerate(state_action) if value == max(state_action)])

    def learn(self, state, action, reward, next_state, next_action):
        """
        The Q value for a state-action is updated by an error, adjusted by 
        the learning rate alpha. 
        Q values represent the possible reward received in the next time step 
        for taking action a in state s, plus the discounted future reward 
        received from the next state-action observation.

        SARSA learns the Q values associated with taking the policy it follows itself
        """
        current_q = self.q_table[state][action]
        next_q = self.q_table[next_state][next_action]
        new_q = (current_q + self.learning_rate *
                 (reward + self.discount_factor * next_q - current_q))
        self.q_table[state][action] = new_q
