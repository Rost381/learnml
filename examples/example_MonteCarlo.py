import io
import random

import numpy as np
from PIL import Image, ImageTk

from learnml.reinforcement.api import MonteCarlo, MonteCarloEnv


def main():
    env = MonteCarloEnv()
    model = MonteCarlo(actions=list(range(env.n_actions)))

    episodes = 10
    for episode in range(episodes):
        state = env.reset()
        action = model.get_action(state)

        while True:
            env.render()

            """forward to next state. reward is number and done is boolean"""
            next_state, reward, done = env.step(action)
            model.save_sample(next_state, reward, done)

            """get next action"""
            action = model.get_action(next_state)

            if episode == episodes - 1:
                ps = env.saveimage()
                img = Image.open(io.BytesIO(ps.encode('utf-8')))
                # seems works on MAC, not on PC
                img.save('./examples/example_MonteCarlo.png')

            """end of each episode, update the q function table"""
            if done:
                print("episode : ", episode)
                model.update()
                model.samples.clear()
                break


if __name__ == "__main__":
    main()
