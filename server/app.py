from fastapi import FastAPI
from pydantic import BaseModel
from environment import Environment

app = FastAPI()
env = Environment()

# Request schema
class ActionRequest(BaseModel):
    action: int


@app.get("/reset")
def reset():
    return env.reset()


@app.post("/step")
def step(req: ActionRequest):
    return env.step(req.action)


@app.get("/state")
def state():
    return env.state_info()