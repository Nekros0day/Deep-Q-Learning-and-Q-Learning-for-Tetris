import numpy as np
import pickle
import copy
import random
import math
import h5py
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

# This file provides the skeleton structure for the classes TQAgent and TDQNAgent to be completed by you, the student.
# Locations starting with # TO BE COMPLETED BY STUDENT indicates missing code that should be written by you.

class TQAgent:
    # This class defines an agent that uses Q-learning to play Tetris. The agent learns
    # to optimize its actions (moves and rotations of Tetris pieces) to maximize the
    # game score over multiple episodes of gameplay.

    def __init__(self, alpha, epsilon, episode_count):
        """
        Initialize the Q-learning agent with the necessary parameters and state variables.
        
        Args:
        alpha (float): The learning rate which determines the extent to which the newly acquired
                       information will override the old information.
        epsilon (float): The exploration rate which defines the probability of agent choosing
                         a random action over the best action according to the Q-table.
        episode_count (int): The total number of episodes that the agent will be trained across.
        
        Attributes:
        alpha (float): Learning rate.
        epsilon (float): Exploration rate for epsilon-greedy policy.
        episode (int): Counter to track the current episode number.
        episode_count (int): Total number of episodes for training.
        cumulative_rewards (list): List to store the cumulative rewards after each episode.
        """
        self.alpha = alpha
        self.epsilon = epsilon
        self.episode = 0
        self.episode_count = episode_count
        self.cumulative_rewards = []

    def fn_init(self, gameboard):
        """
        Initialize the agent's state and action spaces, as well as the Q-table based on the gameboard configuration.

        Args:
        gameboard (TGameBoard): The gameboard instance which provides the structural details of the Tetris game such as number of rows and columns, and tile types.
        
        Initialization:
        Initializes the Q-table with all zeros. The Q-table's dimensions are determined by the number of possible states and actions.
        States are represented by a binary encoding of the gameboard grid combined with the identifier for the upcoming tile.
        Actions are represented as tuples of (position, rotation) indicating where and how tiles can be placed.
        """
        self.gameboard = gameboard
        positions = gameboard.N_col  # Number of horizontal positions each piece can be placed in.
        rotations = 4  # Four possible rotations (0, 90, 180, 270 degrees).

        # State representation combining the gameboard's binary occupancy grid and the upcoming tile type.
        self.currentState = np.zeros(gameboard.N_row * gameboard.N_col + len(gameboard.tiles))

        # Generate all possible actions based on the number of positions and rotations.
        self.actions = [(i, j) for i in range(positions) for j in range(rotations)]
        self.actions = np.array(self.actions)

        # Initialize the Q-table, each row corresponds to a state and each column to an action.
        self.Qtable = np.zeros((2**(gameboard.N_row * gameboard.N_col + len(gameboard.tiles)), len(self.actions)))
        
        # Initialize a reward storage array for tracking rewards received in each episode.
        self.rewards = np.zeros(self.episode_count)

    def fn_load_strategy(self, strategy_file):
        """
        Load a pre-trained strategy (Q-table) from a file to evaluate the agent's performance with an existing strategy.
        
        Args:
        strategy_file (str or file-like object): The file path or file-like object from which the Q-table will be loaded.
        """
        self.Qtable = np.loadtxt(strategy_file, delimiter=',')

    def fn_read_state(self):
        """
        Converts the current game board and the next tile type into a unique integer that represents
        the current state in the Q-table. This function handles both the binary conversion of the game board
        and mapping of the current tile type into the state representation.
        
        Implementation:
        - Flattens the 2D game board array into a 1D array.
        - Converts the game board's elements from -1 to 0 to ensure a binary representation (0s and 1s).
        - Prepares a one-hot encoded array for the current tile type indicating which tile is currently active.
        - Concatenates the binary game board array with the one-hot encoded tile array to form a complete state vector.
        - Converts this binary vector into a string and then into a unique integer (by interpreting the string as a binary number),
          which efficiently represents the current state as an index for the Q-table.
        """
        currentBoard = np.ndarray.flatten(self.gameboard.board)  # Flatten the 2D game board to a 1D array.
        binaryCurrentBoard = np.where(currentBoard == -1, 0, currentBoard)  # Convert -1s to 0s for binary representation.

        nextTile = np.zeros(len(self.gameboard.tiles))  # Initialize a zero array for one-hot encoding of the tile type.
        nextTile[self.gameboard.cur_tile_type] = 1  # Set the current tile type index to 1.

        # Combine the binary board representation with the one-hot tile type.
        binaryCurrentState = np.concatenate((binaryCurrentBoard, nextTile))
        # Convert binary array to a binary string and then to an integer index.
        binaryCurrentStateString = ''.join(str(int(x)) for x in binaryCurrentState)
        self.currentStateIndex = int(binaryCurrentStateString, 2)  # Store the state index for accessing the Q-table.

    def fn_select_action(self):
        """
        Selects an action using an epsilon-greedy strategy where the agent decides either to explore a new action randomly
        or to exploit the best-known action according to the Q-table. Executes the selected action by interfacing with the game board.

        Implementation:
        - Generates a random number and compares it with epsilon to decide between exploration and exploitation.
        - If exploring, randomly select an action until a valid move is found (game board accepts the move).
        - If exploiting, selects the action with the highest Q-value; if there are ties, one is chosen at random.
        - Tries to execute the selected action using the game board's movement function.
        - Marks illegal moves in the Q-table with negative infinity to prevent future selection.
        """
        hasMoved = False
        r = np.random.uniform(0, 1)  # Generate a random number to compare against epsilon.
        if r < self.epsilon:
            while not hasMoved:
                self.currentActionIndex = np.random.randint(0, len(self.actions))  # Randomly select an action.
                move = self.gameboard.fn_move(self.actions[self.currentActionIndex][0], self.actions[self.currentActionIndex][1])
                if move == 0:
                    hasMoved = True  # Move was successful.
                else:
                    self.Qtable[self.currentStateIndex, self.currentActionIndex] = -np.inf  # Penalize illegal moves.
        else:
            while not hasMoved:
                # Select the action with the maximum Q-value at the current state.
                maxQIndices = np.where(self.Qtable[self.currentStateIndex, :] == np.max(self.Qtable[self.currentStateIndex, :]))[0]
                self.currentActionIndex = np.random.choice(maxQIndices) if len(maxQIndices) > 1 else maxQIndices[0]

                move = self.gameboard.fn_move(self.actions[self.currentActionIndex][0], self.actions[self.currentActionIndex][1])
                if move == 0:
                    hasMoved = True  # Move was successful.
                else:
                    self.Qtable[self.currentStateIndex, self.currentActionIndex] = -np.inf  # Penalize illegal moves.
    
    def fn_reinforce(self, old_state, reward):
        """
        Updates the Q-table based on the outcome of an action taken. This is the core of the learning
        mechanism in Q-learning where we adjust the estimated Q-values based on the reward received after
        taking an action and observing the new state.

        Args:
        old_state (int): Index of the state from which the action was taken.
        reward (float): Reward received after taking the action.

        Implementation:
        - Retrieves the action index that was most recently taken.
        - Applies the Q-learning formula to update the Q-value for the state-action pair.
          The update is a blend of the old value and the learned value (reward + discounted maximum future Q-value).
        """
        old_action = self.currentActionIndex  # Retrieve the last action taken.
        # Update the Q-value using the Q-learning update rule.
        self.Qtable[old_state, old_action] += self.alpha * (reward + np.max(self.Qtable[self.currentStateIndex, :]) - self.Qtable[old_state, old_action])

    def fn_turn(self):
        """
        Handles the logic for one turn in the game. This includes checking if the game is over, selecting
        and executing actions, updating the state, and reinforcing the learning based on the action's outcome.

        Implementation:
        - Checks if the game is over; if so, processes end-of-episode logic including logging, plotting, and resetting.
        - If the game is not over, it proceeds to select an action, execute it, observe the reward, update the game state,
          and finally update the Q-table based on the action's outcome.
        """
        if self.gameboard.gameover:
            # End of episode: log cumulative reward and update plots.
            self.cumulative_rewards.append(self.rewards[self.episode])
            self.episode += 1

            # Periodically log progress.
            if self.episode % 100 == 0:
                print(f'Episode {self.episode}/{self.episode_count} (reward: {np.sum(self.rewards[self.episode-100:self.episode])})')


            # Optionally save data at certain episodes.
            if self.episode % 1000 == 0:
                if self.episode in [1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 500000, 1000000]:
                    self.plot_rewards()
                    np.savetxt(f"Rewards_x_{self.episode}.csv", self.rewards)
                    # Save the Q-table when the training is complete.
                    np.savetxt(f"X_Qtable_final_{self.episode}.csv", self.Qtable, delimiter=",")

            # Check if training is complete.
            if self.episode >= self.episode_count:
                raise SystemExit(0)
            else:
                self.gameboard.fn_restart()
        else:
            # Regular game turn: select and execute action, update the state and rewards.
            self.fn_select_action()

            old_state = self.currentStateIndex  # Store current state to use for learning update.

            # Execute the action and receive the reward.
            reward = self.gameboard.fn_drop()
            self.rewards[self.episode] += reward  # Accumulate rewards for the current episode.

            self.fn_read_state()  # Read the new state after the action.
            self.fn_reinforce(old_state, reward)  # Update the Q-table using the reward and transition.
            
            
    def plot_rewards(self):
        # Plot the returns and the moving average over 100 episodes
        plt.figure(figsize=(10, 5))
        plt.plot(self.cumulative_rewards, label='Return per Episode')
        if len(self.cumulative_rewards) >= 100:
            moving_avg = np.convolve(self.cumulative_rewards, np.ones((100,))/100, mode='valid')
            plt.plot(np.arange(99, len(self.cumulative_rewards)), moving_avg, label='Moving Average (100 episodes)')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title('Return and Moving Average of Returns Over Training Episodes')
        plt.legend()
        plt.savefig(f"plot_{len(self.cumulative_rewards)}.png")  # Save the plot file
        plt.close()

class QN(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(QN, self).__init__()
        self.fc1 = nn.Linear(in_dim, 64, dtype=torch.float64)
        self.fc2 = nn.Linear(64, 64, dtype=torch.float64)
        self.fc3 = nn.Linear(64, out_dim, dtype=torch.float64)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class TDQNAgent:
    # Initializes a Deep Q-Network agent designed to play Tetris using Q-learning methods.
    def __init__(self, alpha, epsilon, epsilon_scale, replay_buffer_size, batch_size, sync_target_episode_count, episode_count):
        """
        Initializes the TDQNAgent with specified parameters.
        
        """
        self.alpha = alpha  # Learning rate for the Adam optimizer.
        self.epsilon = epsilon  # Initial exploration rate.
        self.epsilon_scale = epsilon_scale  # Factor to decrease epsilon, reducing exploration over episodes.
        self.replay_buffer_size = replay_buffer_size  # Maximum size of the experience replay buffer.
        self.batch_size = batch_size  # Number of experiences to sample from buffer when learning.
        self.sync_target_episode_count = sync_target_episode_count  # Episodes between synchronization of target and policy networks.
        self.episode = 0  # Initialize current episode count.
        self.episode_count = episode_count  # Total number of episodes to train for.
        self.reward_tots = [0] * episode_count  # Initialize a list to store total rewards per episode.

    def fn_init(self, gameboard):
        """
        Initializes the agent with the game board and prepares neural networks for action prediction and evaluation.

        Args:
            gameboard (GameBoard): The game board instance with which the agent will interact.
        """
        self.gameboard = gameboard  # Store the gameboard instance.
        self.actions = []
        # Initialize the policy Q-network with input size based on gameboard dimensions and number of actions, and output size equal to possible action choices.
        self.qn = QN(gameboard.N_col * gameboard.N_row + len(gameboard.tiles), 16)
        self.qnhat = copy.deepcopy(self.qn)  # Create a deep copy of the policy network to serve as the target network.
        self.exp_buffer = []  # Initialize the experience replay buffer.
        self.criterion = nn.MSELoss()  # Set the loss function to Mean Squared Error.
        self.optimizer = torch.optim.Adam(self.qn.parameters(), lr=self.alpha)  # Initialize the optimizer with the learning rate.

    def fn_load_strategy(self, strategy_file):
        """
        Loads a pre-trained strategy from a file into the policy network.

        Args:
            strategy_file (str): Path to the file containing the saved model state.
        """
        self.qn.load_state_dict(torch.load(strategy_file))  # Load the saved network weights into the policy network.


    def fn_read_state(self):
        """
        Reads the current state of the game board and encodes it for the neural network.

        This method flattens the 2D game board array to a 1D array and appends the current
        tile type indicator to form a complete state representation.
        """
        # Flatten the 2D game board into a 1D array for neural network input.
        self.state = self.gameboard.board.flatten()

        # Initialize an array of -1s with a length equal to the number of tile types.
        tile_type = -np.ones(len(self.gameboard.tiles))

        # Set the index corresponding to the current tile type to 1, indicating the presence of this tile type.
        tile_type[self.gameboard.cur_tile_type] = 1

        # Append the tile type array to the flattened game board array to form the complete state.
        self.state = np.append(self.state, tile_type)

    def fn_select_action(self):
        """
        Selects and executes an action using the epsilon-greedy strategy based on the current state.

        This function decides whether to take a random action or one based on the neural network's output,
        depending on the epsilon value.
        """
        # Switch the policy network to evaluation mode to prevent training updates during action selection.
        self.qn.eval()

        # Compute the Q-values for the current state and detach from the computation graph.
        out = self.qn(torch.tensor(self.state)).detach().numpy()

        # Epsilon-greedy choice: decide whether to explore or exploit.
        if np.random.rand() < max(self.epsilon, 1 - self.episode / self.epsilon_scale):
            # Exploration: choose a random action index.
            self.action = random.randint(0, 15)
        else:
            # Exploitation: choose the action with the highest Q-value.
            self.action = np.argmax(out)

        # Decode the selected action to get the rotation and position.
        rotation = int(self.action / 4)
        position = self.action % 4

        # Execute the chosen action on the game board.
        self.gameboard.fn_move(position, rotation)

    def fn_reinforce(self, batch):
        """
        Updates the Q-network based on a batch of transitions.

        This method calculates the loss between predicted Q-values by the policy network and
        the target Q-values calculated using the target network. It then updates the policy network
        using gradient descent to minimize this loss.

        Args:
            batch (list of tuples): Each tuple contains (state, action, reward, next_state, terminal)
                                representing a transition.
        """
        targets = []  # List to hold the target Q-values for each transition.
        action_value = []  # List to hold the Q-values predicted for the taken actions.

        # Set the policy network to training mode and target network to evaluation mode.
        self.qn.train()
        self.qnhat.eval()

        # Process each transition in the batch.
        for transition in batch:
            state, action, reward, next_state, terminal = transition

            # Start with the immediate reward.
            y = reward

            # If the episode did not end, add the maximum predicted Q-value of the next state.
            if not terminal:
                out = self.qnhat(torch.tensor(next_state)).detach().numpy()  # Q-values from target network.
                y += max(out)  # Add the maximum Q-value to the immediate reward.

            # Append the calculated target Q-value to the list.
            targets.append(torch.tensor(y, dtype=torch.float64))

            # Predict the Q-value for the current state and the action taken.
            out = self.qn(torch.tensor(state))
            action_value.append(out[action])

        # Convert lists to tensors.
        targets = torch.stack(targets)
        action_value = torch.stack(action_value)

        # Calculate the mean squared error loss.
        loss = self.criterion(action_value, targets)

        # Perform gradient descent.
        self.optimizer.zero_grad()  # Reset gradients to zero.
        loss.backward()  # Compute gradient of the loss.
        self.optimizer.step()  # Update the network weights.

    def fn_turn(self):
        """
        Handles the logic for each turn of the game.

        This method checks if the game is over, updates the episode count, possibly saves
        the current network state, and manages transitions within the game.
        """
        if self.gameboard.gameover:
            # Increment the episode count since the game is over.
            self.episode += 1

            # Print the average reward every 100 episodes.
            if self.episode % 100 == 0:
                print('Episode {}: Average reward over the last 100 episodes: {}'.format(
                    self.episode, np.mean(self.reward_tots[self.episode-100:self.episode])))

            # Save the network state at specified episodes.
            if self.episode % 1000 == 0 and self.episode in [1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 500000, 1000000]:
                torch.save(self.qn.state_dict(), 'qn_{}.pth'.format(self.episode))
                torch.save(self.qnhat.state_dict(), 'qnhat_{}.pth'.format(self.episode))
                pickle.dump(self.reward_tots, open('reward_tots_{}.p'.format(self.episode), 'wb'))

            # End training if the maximum episode count is reached.
            if self.episode >= self.episode_count:
                raise SystemExit(0)

            # Sync the target network periodically.
            if len(self.exp_buffer) >= self.replay_buffer_size and (self.episode % self.sync_target_episode_count) == 0:
                self.qnhat = copy.deepcopy(self.qn)

            # Restart the game board for a new game.
            self.gameboard.fn_restart()

        else:
            # Continue with normal gameplay.
            self.fn_select_action()
            old_state = self.state.copy()
            reward = self.gameboard.fn_drop()
            self.reward_tots[self.episode] += reward
            self.fn_read_state()
            self.exp_buffer.append((old_state, self.action, reward, self.state.copy(), self.gameboard.gameover))

            # Learn from experiences if the buffer is full.
            if len(self.exp_buffer) >= self.replay_buffer_size:
                batch = random.sample(self.exp_buffer, k=self.batch_size)
                self.fn_reinforce(batch)
                if len(self.exp_buffer) >= self.replay_buffer_size + 2:
                    self.exp_buffer.pop(0)  # Remove the oldest transition.



class THumanAgent:
    def fn_init(self,gameboard):
        self.episode=0
        self.reward_tots=[0]
        self.gameboard=gameboard

    def fn_read_state(self):
        pass

    def fn_turn(self,pygame):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit(0)
            if event.type==pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.reward_tots=[0]
                    self.gameboard.fn_restart()
                if not self.gameboard.gameover:
                    if event.key == pygame.K_UP:
                        self.gameboard.fn_move(self.gameboard.tile_x,(self.gameboard.tile_orientation+1)%len(self.gameboard.tiles[self.gameboard.cur_tile_type]))
                    if event.key == pygame.K_LEFT:
                        self.gameboard.fn_move(self.gameboard.tile_x-1,self.gameboard.tile_orientation)
                    if event.key == pygame.K_RIGHT:
                        self.gameboard.fn_move(self.gameboard.tile_x+1,self.gameboard.tile_orientation)
                    if (event.key == pygame.K_DOWN) or (event.key == pygame.K_SPACE):
                        self.reward_tots[self.episode]+=self.gameboard.fn_drop()