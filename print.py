import pickle
import matplotlib.pyplot as plt
import numpy as np

# Load data from the pickle file
with open('reward_tots_10000.p', 'rb') as file:
    cumulative_rewards = pickle.load(file)

# Create the plot with specified dimensions and style
plt.figure(figsize=(10, 5))
plt.plot(cumulative_rewards, label='Return per Episode')

# Calculate and plot the moving average if there are enough data points
if len(cumulative_rewards) >= 100:
    moving_avg = np.convolve(cumulative_rewards, np.ones((100,))/100, mode='valid')
    plt.plot(np.arange(99, len(cumulative_rewards)), moving_avg, label='Moving Average (100 episodes)')

# Set labels, title, and show grid
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Return and Moving Average of Returns Over Training Episodes')

# Add legend and grid
plt.legend()
plt.grid(True)

# Save the plot
plt.savefig(f"plot_{len(cumulative_rewards)}.png")
plt.close()  # Close the plot to free up memory
