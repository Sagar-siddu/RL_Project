## Project Overview

This project demonstrates two different approaches to implementing a Reinforcement Learning (RL) environment and agent interaction.

---

## Execution Methods

### 1. Single File Execution (`test_env.py`)

In this approach, the entire implementation is contained within a single file.

* Environment, agent logic, and execution loop are tightly coupled
* Simple and easy to understand
* Suitable for learning and quick experimentation

**Limitations:**

* Not scalable
* Difficult to extend or maintain for larger systems
* Cannot support distributed or parallel execution

---

### 2. Modular API-Based Execution

In this approach, the project is divided into multiple components:

* **Environment (`environment.py`)**
  Contains the core environment logic (state, actions, rewards)

* **API Layer (`app.py`)**
  Wraps the environment using a FastAPI server and exposes endpoints such as `/reset`, `/step`, and `/state`

* **Agent (`run_agent.py`)**
  Interacts with the environment via HTTP requests and determines actions

---

## Why This Approach is Preferred

This modular design provides several advantages:

### 1. Scalability

Multiple agents can interact with the same environment concurrently, enabling parallel training.

### 2. Distributed Execution

The environment can run on a separate machine (e.g., a server handling heavy computation), while the agent runs locally or elsewhere.

### 3. Language Independence

The agent and environment are decoupled and can be implemented in different programming languages, communicating via APIs.

### 4. Separation of Concerns

Each component (environment, API, agent) is independent, making the system easier to maintain, debug, and
