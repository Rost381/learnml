import time
import tkinter as tk
import io
import numpy as np
from PIL import Image, ImageTk

np.random.seed(2)
PhotoImage = ImageTk.PhotoImage
TITLE = 'Q-Learning'
BG_COLOR = 'white'
LINE_COLOR = '#D8D8D8'
UNIT = 100
HEIGHT = 4
WIDTH = 4
IMG_PATH = 'alphalearn/reinforcement/img/'


class QLearningEnv(tk.Tk):
    def __init__(self):
        super(QLearningEnv, self).__init__()
        self.action_space = ['u', 'd', 'l', 'r']
        self.n_actions = len(self.action_space)
        self.title(TITLE)
        self.geometry('{0}x{1}'.format(HEIGHT * UNIT, HEIGHT * UNIT))
        self.shapes = self._load_images()
        self.canvas = self._build_canvas()
        self.texts = []

    def _build_canvas(self):
        canvas = tk.Canvas(self,
                           bg=BG_COLOR,
                           height=HEIGHT * UNIT,
                           width=WIDTH * UNIT)
        """grids"""
        for c in range(0, WIDTH * UNIT, UNIT):
            x0, y0, x1, y1 = c, 0, c, HEIGHT * UNIT
            canvas.create_line(x0, y0, x1, y1, fill=LINE_COLOR)
        for r in range(0, HEIGHT * UNIT, UNIT):
            x0, y0, x1, y1 = 0, r, HEIGHT * UNIT, r
            canvas.create_line(x0, y0, x1, y1, fill=LINE_COLOR)

        """add images to canvas"""
        self.ball = canvas.create_image(50, 50, image=self.shapes[0])
        self.wall1 = canvas.create_image(250, 150, image=self.shapes[1])
        self.wall2 = canvas.create_image(150, 250, image=self.shapes[1])
        self.goal = canvas.create_image(250, 250, image=self.shapes[2])

        canvas.pack()

        return canvas

    def _load_images(self):
        ball = PhotoImage(
            Image.open(IMG_PATH + "ball.png").resize((65, 65)))
        wall = PhotoImage(
            Image.open(IMG_PATH + "wall.png").resize((65, 65)))
        goal = PhotoImage(
            Image.open(IMG_PATH + "goal.png").resize((65, 65)))

        return ball, wall, goal

    def text_value(self, row, col, contents, action, font='Helvetica', size=9,
                   style='normal', anchor="nw"):

        if action == 0:
            origin_x, origin_y = 7, 42
        elif action == 1:
            origin_x, origin_y = 85, 42
        elif action == 2:
            origin_x, origin_y = 42, 5
        else:
            origin_x, origin_y = 42, 77

        x, y = origin_y + (UNIT * col), origin_x + (UNIT * row)
        font = (font, str(size), style)
        text = self.canvas.create_text(x, y, fill="black", text=contents,
                                       font=font, anchor=anchor)
        return self.texts.append(text)

    def print_value_all(self, q_table):
        for i in self.texts:
            self.canvas.delete(i)
        self.texts.clear()
        for i in range(HEIGHT):
            for j in range(WIDTH):
                for action in range(0, 4):
                    state = [i, j]
                    if str(state) in q_table.keys():
                        temp = q_table[str(state)][action]
                        self.text_value(j, i, round(temp, 2), action)

    def coords_to_state(self, coords):
        x = int((coords[0] - 50) / 100)
        y = int((coords[1] - 50) / 100)
        return [x, y]

    def state_to_coords(self, state):
        x = int(state[0] * 100 + 50)
        y = int(state[1] * 100 + 50)
        return [x, y]

    def reset(self):
        self.update()
        time.sleep(0.5)
        x, y = self.canvas.coords(self.ball)
        self.canvas.move(self.ball, UNIT / 2 - x, UNIT / 2 - y)
        self.render()
        return self.coords_to_state(self.canvas.coords(self.ball))

    def step(self, action):
        state = self.canvas.coords(self.ball)
        base_action = np.array([0, 0])
        self.render()

        if action == 0:  # up
            if state[1] > UNIT:
                base_action[1] -= UNIT
        elif action == 1:  # down
            if state[1] < (HEIGHT - 1) * UNIT:
                base_action[1] += UNIT
        elif action == 2:  # left
            if state[0] > UNIT:
                base_action[0] -= UNIT
        elif action == 3:  # right
            if state[0] < (WIDTH - 1) * UNIT:
                base_action[0] += UNIT

        """move agent"""
        self.canvas.move(self.ball, base_action[0], base_action[1])
        """move ball to top level of canvas"""
        self.canvas.tag_raise(self.ball)
        next_state = self.canvas.coords(self.ball)

        """reward function
        if goal: 100
        if wall: -100
        if blank: keep runing
        """
        if next_state == self.canvas.coords(self.goal):
            reward = 100
            done = True
        elif next_state in [self.canvas.coords(self.wall1),
                            self.canvas.coords(self.wall2)]:
            reward = -100
            done = True
        else:
            reward = 0
            done = False

        next_state = self.coords_to_state(next_state)
        return next_state, reward, done

    def render(self):
        time.sleep(0.03)
        self.update()

    def saveimage(self):
        ps = self.canvas.postscript(colormode='color')
        return ps
