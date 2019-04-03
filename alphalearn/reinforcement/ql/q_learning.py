import random
from collections import defaultdict

import numpy as np


class QLearning():
    """Q-learning is a model-free reinforcement learning algorithm.

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
        a number between 0 and 1 and has the effect of valuing rewards received earlier
        higher than those received later. reflecting the value of a "good start".
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
        max_index_list = []
        max_value = state_action[0]
        for index, value in enumerate(state_action):
            if value > max_value:
                max_index_list.clear()
                max_value = value
                max_index_list.append(index)
            elif value == max_value:
                max_index_list.append(index)
        return random.choice(max_index_list)

    def learn(self, state, action, reward, next_state):
        """Steps 3: Update q table

        Bellman Optimality Equation
        new_q : new Q value for that state and the action.
        current_q : current Q value for that state and the action.
        max(self.q_table[next_state]): maximum expected future reward.
        reward : reward for taking that action at that state.
        """
        current_q = self.q_table[state][action]
        new_q = reward + self.discount_factor * max(self.q_table[next_state])
        self.q_table[state][action] += self.learning_rate * (new_q - current_q)
