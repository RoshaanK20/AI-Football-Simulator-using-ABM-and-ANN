# AI Based Football Training and Match Strategy Simulator using ANN and ABM

A beginner friendly Artificial Intelligence semester project that combines Artificial Neural Networks and Agent Based Modelling to simulate football player behavior, strategy prediction, and match visualization.

---

# Project Overview

This project demonstrates how AI can be applied in football simulations using:

* Artificial Neural Networks for pass success prediction
* Agent Based Modelling for player behavior simulation
* Pygame for football field visualization and real time match simulation

The simulator creates a simplified football environment where players act independently, move around the field, pass the ball, lose stamina, and make decisions based on match conditions.

---

# Features

## Football Match Simulation

* Real time football field visualization
* Two football teams
* Ball movement simulation
* Match timer and scoreboard
* Simple football gameplay mechanics

## Player Agents

Each football player acts as an independent agent.

Player abilities include:

* Moving around the field
* Following the ball
* Passing
* Defending
* Dribbling
* Fatigue and stamina reduction

## Artificial Neural Network

The ANN predicts:

* Pass success probability
* Basic strategy recommendation

ANN Inputs:

* Stamina
* Passing accuracy
* Nearby defenders
* Distance from goal
* Match minute

## Match Statistics

The simulator displays:

* Match time
* Team scores
* Ball possession
* Player stamina
* ANN strategy recommendations
* Pass success probability

## Visualization

Pygame is used to visualize:

* Green football field
* Red and blue teams
* White football
* Match statistics
* Strategy suggestions

---

# Technologies Used

| Technology         | Purpose                   |
| ------------------ | ------------------------- |
| Python             | Main programming language |
| TensorFlow / Keras | Artificial Neural Network |
| NumPy              | Numerical operations      |
| Pandas             | Dataset handling          |
| Pygame             | Football visualization    |
| Matplotlib         | Data visualization        |
| Scikit learn       | Data preprocessing        |

---

# Project Structure

```text
football_ai_simulator/
│
├── main.py
├── dataset_generator.py
├── ann_model.py
├── player_agent.py
├── football_field.py
├── strategy_system.py
├── football_dataset.csv
├── football_ann_model.h5
└── README.md
```

---

# Installation

## Clone Repository

```bash
git clone https://github.com/yourusername/football-ai-simulator.git
cd football-ai-simulator
```

## Install Dependencies

```bash
pip install tensorflow numpy pandas pygame matplotlib scikit-learn
```

---

# How to Run

Run the main file:

```bash
python main.py
```

The system will automatically:

1. Generate synthetic football dataset
2. Train the ANN model
3. Save the trained model
4. Start football simulation

---

# Artificial Neural Network

The project uses a simple Feedforward Neural Network.

## ANN Architecture

Input Layer

↓

Dense Layer with ReLU Activation

↓

Dense Layer with ReLU Activation

↓

Sigmoid Output Layer

## ANN Details

| Component           | Value               |
| ------------------- | ------------------- |
| Network Type        | Feedforward ANN     |
| Activation Function | ReLU                |
| Output Function     | Sigmoid             |
| Optimizer           | Adam                |
| Loss Function       | Binary Crossentropy |

---

# Agent Based Modelling

In this project, every football player acts as an autonomous agent.

Each agent has:

* Position
* Team
* Stamina
* Passing accuracy
* Movement behavior
* Ball possession state

Agents interact with:

* Football
* Teammates
* Opponents
* Match environment

---

# System Workflow

```text
START PROJECT
       |
       v
Generate Football Dataset
       |
       v
Train ANN Model
       |
       v
Create Player Agents
       |
       v
Start Match Simulation
       |
       v
Player and Ball Movement
       |
       v
ANN Strategy Prediction
       |
       v
Display Match Statistics
       |
       v
END MATCH
```

---

# System Architecture

```text
+--------------------------+
| Synthetic Dataset Module |
+------------+-------------+
             |
             v
+--------------------------+
| ANN Training Module      |
+------------+-------------+
             |
             v
+--------------------------+
| Strategy Prediction      |
+------------+-------------+
             |
             v
+--------------------------+
| Football Simulation      |
| Player Agents            |
| Ball Movement            |
+------------+-------------+
             |
             v
+--------------------------+
| Pygame Visualization     |
+--------------------------+
```

---

# Educational Purpose

This project is designed for:

* Artificial Intelligence semester projects
* Beginner level AI learning
* ANN understanding
* Agent Based Modelling understanding
* Sports simulation learning
* Viva presentations and demonstrations

---

# Future Improvements

Possible future enhancements include:

* Goalkeeper AI
* Reinforcement Learning
* Tactical formations
* Heatmap analysis
* Real football datasets
* Multiplayer support
* Advanced football physics

---

# Viva Explanation

## Why ANN?

ANN is used to predict pass success probability using football related match conditions.

## Why ABM?

ABM is used because football players behave independently and interact with each other continuously during the match.

## What is AI in this project?

The AI component includes:

* ANN based pass prediction
* Agent based player behavior
* Strategy recommendation system

---

# Author

## Roshaan Ahmed Khan

BS Artificial Intelligence
BAI-24S-002

Instructor: Sir Tayyab

---

# License

This project is created for educational and academic purposes.
