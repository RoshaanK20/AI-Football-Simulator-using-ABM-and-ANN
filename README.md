# AI Based Agent Based Modeling System

A beginner friendly Agent Based Modeling (ABM) simulation developed in Python and Mesa.
This project demonstrates how multiple agents interact inside a 2D grid environment using simple rules such as movement, eating, resting, and energy management.

The project includes:

* Pure Python ABM implementation
* Mesa framework implementation
* Agent interaction simulation
* Grid based environment
* Energy based behaviors
* Multi agent movement and interaction

---

# What is Agent Based Modeling?

Agent Based Modeling (ABM) is a simulation technique where multiple independent entities called agents interact inside an environment.

Each agent has:

## State

The internal condition of the agent.

Example:

```python
energy = 20
position = (2, 3)
```

## Behavior

Actions that agents can perform.

Example:

```python
move()
eat()
rest()
```

## Environment

The world where agents exist and interact.

In this project, the environment is a 2D grid.

---

# Types of Agents

## Reactive Agents

Agents that respond directly to conditions.

Example:

* Rest when energy is low
* Move when energy is sufficient

## Goal Based Agents

Agents that attempt to achieve objectives.

Example:

* Survive as long as possible
* Search for food

## Behavior Based Agents

Agents that combine multiple simple behaviors.

Example:

* Move
* Eat
* Rest
* Avoid occupied cells

---

# Features

## Pure Python Implementation

* Manual agent handling
* 2D grid system
* Random movement
* Food system
* Energy management
* Agent interaction
* Cell occupancy rules
* Simple visualization

## Mesa Implementation

* Structured agent management
* MultiGrid environment
* Agent scheduling
* Step based simulation
* Easier scalability

---

# Technologies Used

* Python
* NumPy
* Mesa
* Random Module

---

# Project Structure

```text
├── ABM_updated.py
├── ABM1.py
└── README.md
```

---

# How the Simulation Works

1. Agents are placed randomly in the grid
2. Each agent starts with energy
3. During every simulation step:

   * Agents move
   * Agents eat food
   * Agents rest if energy is low
4. Energy decreases after movement
5. Agents gain energy after eating
6. Agents interact with the environment and other agents

---

# Pure Python ABM

The pure Python version manually handles:

* Grid management
* Agent movement
* Collision checking
* Food handling
* Simulation updates

Run:

```bash
python ABM_updated.py
```

---

# Mesa ABM

Mesa simplifies:

* Agent scheduling
* Grid management
* Environment handling
* Agent interaction

Run:

```bash
python ABM1.py
```

Install Mesa:

```bash
pip install mesa
```

---

# Example Concepts Used

## Grid Environment

The environment is represented using a matrix:

```text
. = Empty Cell
F = Food
A = Agent
```

---

# Adjustable Parameters

You can modify:

```python
grid_size
number_of_agents
food_count
energy
simulation_steps
```

Changing these parameters affects the behavior of the system.

---

# Comparison of Approaches

| Approach                       | Advantages                       | Limitations                 |
| ------------------------------ | -------------------------------- | --------------------------- |
| Pure Python                    | Full control and flexibility     | More coding required        |
| Mesa                           | Easier structure and scalability | Requires external framework |
| Visualization Based Simulation | Easy to understand behavior      | Additional setup needed     |

---

# Learning Outcomes

This project demonstrates:

* Fundamentals of Agent Based Modeling
* Agent behavior simulation
* Grid based environments
* Multi agent systems
* Rule based interaction
* Mesa framework basics
* Simulation design concepts

---

# Future Improvements

Possible future enhancements:

* Smarter agents
* Pathfinding
* Reproduction system
* Predator prey simulation
* Graphical interface
* Real time visualization
* Reinforcement learning integration

---

# Author

Roshaan Ahmed Khan
BS Artificial Intelligence
SMIU
