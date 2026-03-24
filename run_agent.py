import requests
import random

BASE_URL = "http://127.0.0.1:8000"

# Reset environment
res = requests.get(f"{BASE_URL}/reset")
data = res.json()

done = data["done"]
total_score = 0

while not done:
    action = random.choice([0,1,2,3])

    res = requests.post(f"{BASE_URL}/step", json={"action": action})
    data = res.json()

    state = data["observation"]
    reward = data["reward"]
    done = data["done"]

    total_score += reward
    print(state, reward)

print(f"\nTotal Score: {total_score}")
print("Positive" if total_score > 0 else "Negative" if total_score < 0 else "Neutral")