import random

class GridWorld:

    def __init__(self):
        self.size = 5
        self.goal = (4, 4)
        self.state = (0, 0)

    def reset(self):
        self.state = (0, 0)
        return self.state

    def step(self, action):
        x, y = self.state

        if action == 0:  # up
            x -= 1
        elif action == 1:  # down
            x += 1
        elif action == 2:  # left
            y -= 1
        elif action == 3:  # right
            y += 1

        # boundary check
        x = max(0, min(x, 4))
        y = max(0, min(y, 4))

        self.state = (x, y)

        # reward
        if self.state == self.goal:
            return self.state, 10, True, {}
        else:
            return self.state, -1, False, {}
        
env = GridWorld()
state = env.reset()

done = False
total_score = 0

while not done:
    action = random.choice([0,1,2,3])
    state, reward, done, _ = env.step(action)
    total_score += reward
    print(state, reward)

print(f"\nTotal Score: {total_score}")
print("Positive" if total_score > 0 else "Negative" if total_score < 0 else "Neutral")        