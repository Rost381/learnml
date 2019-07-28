# Q-learning
Q-learning is a model-free reinforcement learning algorithm. The goal of Q-learning is to learn a policy, which tells an agent what action to take under what circumstances

## Algorithm

$$
Q^{n e w}\left(s_{t}, a_{t}\right) \leftarrow(1-\alpha) \cdot Q\left(s_{t}, a_{t}\right) + lr \cdot \left( \begin{array}{cccc}reward + {\gamma} {\cdot} {\max Q\left(s_{t+1}, a\right)}\end{array}\right)
$$

```python
current_q = self.q_table[state][action]

new_q = reward + self.discount_factor * max(self.q_table[next_state])

self.q_table[state][action] += self.learning_rate * (new_q - current_q)
```