# Deep Q-Learning and Q-Learning for Tetris

This repository contains implementations of Deep Q-Learning (DQN) and Q-Learning algorithms to play the classic game Tetris. The project includes the game logic, the agent classes for Q-Learning and DQN, and a human agent for manual gameplay.

## Overview

The goal of this project is to demonstrate the application of reinforcement learning techniques, specifically Q-Learning and Deep Q-Learning, to the game of Tetris. The agents learn to optimize their actions to maximize the game score over multiple episodes of gameplay.

## Files

- `tetris.py`: Contains the game logic and user interface for Tetris.
- `gameboardclass.py`: Defines the `TGameBoard` class which manages the game state, tile movements, and game rules.
- `agentclass.py`: Contains the agent classes:
  - `TQAgent`: Implements a Q-Learning agent.
  - `TDQNAgent`: Implements a Deep Q-Learning (DQN) agent.
  - `THumanAgent`: Allows manual play through keyboard inputs.

## Q-Learning Agent (TQAgent)

The Q-Learning agent (`TQAgent`) uses a Q-table to learn the optimal policy for playing Tetris. The agent explores the state-action space using an epsilon-greedy strategy and updates the Q-values based on the rewards received from the game environment.

### Key Methods:

- `__init__(self, alpha, epsilon, episode_count)`: Initializes the agent with learning rate, exploration rate, and episode count.
- `fn_init(self, gameboard)`: Initializes the state and action spaces, and the Q-table.
- `fn_load_strategy(self, strategy_file)`: Loads a pre-trained Q-table from a file.
- `fn_read_state(self)`: Converts the current game board state into a unique integer index for the Q-table.
- `fn_select_action(self)`: Selects an action using an epsilon-greedy strategy.
- `fn_reinforce(self, old_state, reward)`: Updates the Q-table based on the reward received.
- `fn_turn(self)`: Handles the logic for one turn in the game and updates the Q-table.

## Deep Q-Learning Agent (TDQNAgent)

The Deep Q-Learning agent (`TDQNAgent`) uses a neural network to approximate the Q-values. The agent employs experience replay and a target network to stabilize training.

### Key Methods:

- `__init__(self, alpha, epsilon, epsilon_scale, replay_buffer_size, batch_size, sync_target_episode_count, episode_count)`: Initializes the agent with necessary parameters for DQN.
- `fn_init(self, gameboard)`: Initializes the neural networks and experience replay buffer.
- `fn_load_strategy(self, strategy_file)`: Loads a pre-trained model from a file.
- `fn_read_state(self)`: Reads the current state of the game board and encodes it for the neural network.
- `fn_select_action(self)`: Selects an action using the epsilon-greedy strategy.
- `fn_reinforce(self, batch)`: Updates the Q-network based on a batch of transitions.
- `fn_turn(self)`: Handles the logic for each turn of the game, including network updates and experience replay.

## Human Agent (THumanAgent)

The human agent (`THumanAgent`) allows a user to play Tetris manually using keyboard inputs.

### Key Methods:

- `fn_init(self, gameboard)`: Initializes the game board for human play.
- `fn_turn(self, pygame)`: Handles keyboard inputs to control the game.

## Dependencies

- numpy
- torch
- matplotlib

Install the dependencies using:
```bash
pip install numpy torch matplotlib
