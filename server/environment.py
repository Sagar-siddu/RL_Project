import numpy as np

class Environment:

    def __init__(self):
        self.size = 5
        self.goal = (4, 4)
        self.state = None
        self.step_count = 0

    def reset(self):
        self.state = (0, 0)
        self.step_count = 0

        return {
            "observation": list(self.state),
            "reward": 0,
            "done": False
        }

    def step(self, action):
        x, y = self.state

        if action == 0:
            x -= 1
        elif action == 1:
            x += 1
        elif action == 2:
            y -= 1
        elif action == 3:
            y += 1

        x = max(0, min(x, self.size - 1))
        y = max(0, min(y, self.size - 1))

        self.state = (x, y)
        self.step_count += 1

        if self.state == self.goal:
            reward = 10
            done = True
        else:
            reward = -1
            done = False

        return {
            "observation": list(self.state),
            "reward": reward,
            "done": done
        }

    def state_info(self):
        return {
            "position": self.state,
            "steps": self.step_count
        }