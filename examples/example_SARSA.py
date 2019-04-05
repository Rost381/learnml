import io

from PIL import Image, ImageTk

from alphalearn.reinforcement.api import SARSA, QLearningEnv


def main():
    env = QLearningEnv()
    model = SARSA(actions=list(range(env.n_actions)))

    episodes = 50
    for episode in range(episodes):
        state = env.reset()
        action = model.choose_action(str(state))

        while True:
            env.render()

            next_state, reward, done = env.step(action)
            next_action = model.choose_action(str(next_state))

            model.learn(str(state), action, reward,
                        str(next_state), next_action)

            state = next_state
            action = next_action

            env.print_value_all(model.q_table)

            if episode == episodes - 1:
                ps = env.saveimage()
                img = Image.open(io.BytesIO(ps.encode('utf-8')))
                # seems works on MAC, not on PC
                img.save('./examples/example_SARSA.png')

            if done:
                break

    print(model.q_table)


if __name__ == "__main__":
    main()
