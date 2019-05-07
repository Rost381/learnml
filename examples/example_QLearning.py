import io

from PIL import Image, ImageTk

from learnml.reinforcement.api import QLearning, QLearningEnv


def main():
    env = QLearningEnv()
    model = QLearning(actions=list(range(env.n_actions)))

    episodes = 50
    for episode in range(episodes):
        state = env.reset()

        while True:
            env.render()

            action = model.choose_action(str(state))
            next_state, reward, done = env.step(action)

            model.learn(str(state), action, reward, str(next_state))

            state = next_state
            env.print_value_all(model.q_table)

            if episode == episodes - 1:
                ps = env.saveimage()
                img = Image.open(io.BytesIO(ps.encode('utf-8')))
                # seems works on MAC, not on PC
                img.save('./examples/example_QLearning.png')

            if done:
                break

    print(model.q_table)


if __name__ == "__main__":
    main()
