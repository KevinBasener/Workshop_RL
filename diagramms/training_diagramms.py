import pandas as pd
import matplotlib.pyplot as plt

# Load the training data from the CSV file
df = pd.read_csv('DQN_1_mit_Auslagerungsbelohnung.csv')

# Create a figure and axis for the plot
plt.figure(figsize=(12, 6))

# Plot the 'Value' (reward) against the 'Step'
plt.plot(df['Step'], df['Value'], label='Reward', alpha=0.5)

# Calculate and plot a rolling mean to show the trend
rolling_mean = df['Value'].rolling(window=10).mean()
plt.plot(df['Step'], rolling_mean, label='Smoothed Reward (Rolling Mean)', color='red')

# Set the title and labels for the plot
plt.title('Reinforcement Learning Training Progress')
plt.xlabel('Step')
plt.ylabel('Reward')

# Add a legend to distinguish the plots
plt.legend()

# Add a grid for better readability
plt.grid(True)

# Ensure the plot starts from the first step value, which is close to zero
plt.xlim(left=0)

# Save the plot to a file
plt.savefig('reward_plot_action_mask.png')

print("Plot saved as reward_plot_action_mask.png")