# AI Football Training and Match Strategy Simulator

This project turns the implementation guide into a runnable Python simulator that combines:

- Agent-Based Modelling ideas for independent football player behavior.
- An Artificial Neural Network for pass, goal, and fatigue prediction.
- A Pygame match view that shows players, the ball, and live strategy recommendations.

## Quick Start

Install dependencies if needed:

```powershell
pip install -r requirements.txt
```

Train the ANN:

```powershell
python main.py --train-only
```

For a direct live demo of only the training step, run:

```powershell
python src\football_ai_simulator\train.py
```

After training, the project saves proof files you can show in your viva:

- `models/training_report.txt` - readable training status, dataset rows, and accuracy.
- `models/training_metrics.json` - raw evaluation metrics.
- `data/training_dataset.csv` - the synthetic training dataset.
- `VIVA_EXPLANATION_GUIDE.txt` - full one-by-one explanation and viva Q&A.

Run the simulator:

```powershell
python main.py
```

Before the Pygame window opens, the terminal asks for:

- Red team final score.
- Blue team final score.
- Red team passes before one goal.
- Blue team passes before one goal.

The simulator then uses those values as match rules. Passes and shots are selected randomly, but a team only scores when it has completed its required number of passes and has not reached the final score you entered. If a player shoots too early, the shot either goes wide of the goal mouth or is blocked by the opponent. Completed passes are counted only when the ball reaches a teammate.

Speed up or slow down the in-game clock:

```powershell
python main.py --match-seconds 60
```

The default is `120`, meaning a full simulated 90-minute match takes about two real minutes.

The first simulator run will train a small model automatically if no saved model exists.
The simulator also shows an `ANN trained` line with accuracy, dataset rows, and MAE near the bottom of the window.

## Project Structure

- `main.py` - simple entry point for training or launching the simulator.
- `src/football_ai_simulator/data.py` - synthetic football dataset generation.
- `src/football_ai_simulator/ann.py` - ANN model, scaler, evaluation, and prediction helpers.
- `src/football_ai_simulator/simulation.py` - agent-based football simulation in Pygame.
- `models/` - generated model files after training.

## Viva Explanation Flow

Start with the big idea:

"Our project combines Agent-Based Modelling and Artificial Neural Networks for football match strategy simulation."

Then explain:

- ABM: each player acts independently with stamina, movement, possession, passing, and shooting behavior.
- ANN: synthetic football data is generated from match features such as stamina, distance from goal, nearby defenders, passing accuracy, and match minute.
- Prediction: the neural network predicts pass success probability, goal probability, and fatigue risk.
- Integration: the simulation sends the current player situation into the ANN, shows the live predictions, and runs randomized passing/shooting under the user-entered score and passes-before-goal rules. Early shots miss wide or give the opponent possession.
