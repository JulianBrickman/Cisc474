import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# Path to your combined results CSV
csv_path = "combined_evaluation_results.csv"  # Update with the actual path to your file

# Load the combined CSV results into a DataFrame
df = pd.read_csv(csv_path)

# Create a new column for the percentage of covered cells
df['covered_cells_percentage'] = df['covered_cells'] / df['coverable_cells'] * 100

# Define model paths and hardcoded reward functions
model_paths = [
    'model_20250331_190457.h5', 'model_20250331_204837.h5', 'model_20250331_222729.h5',  # Yellow (Reward Function 1)
    'model_20250401_011338.h5', 'model_20250401_095150.h5', 'model_20250401_114221.h5', 'model_20250401_134848.h5',  # Blue (Reward Function 2)
    'model_20250401_165803.h5', 'model_20250401_191253.h5', 'model_20250401_234441.h5',  # Red (Reward Function 3)
    'model_20250402_093929.h5', 'model_20250402_124240.h5', 'model_20250402_153117.h5', 'model_20250402_193717.h5',  # Red (Reward Function 3)
    'model_20250403_011102.h5', 'model_20250403_110417.h5', 'model_20250403_212015.h5'  # Orange (Reward Function 4)
]

# Hardcode the colors based on model paths
def assign_color(model_name):
    if model_name in model_paths[:3]:
        return 'yellow', 'Reward Function 1'
    elif model_name in model_paths[3:7]:
        return 'blue', 'Reward Function 2'
    elif model_name in model_paths[7:12]:
        return 'red', 'Reward Function 3'
    elif model_name in model_paths[12:]:
        return 'orange', 'Reward Function 4'
    return 'gray', 'Unknown'

# Apply the categorization to the models
df['color'], df['reward_function'] = zip(*df['model'].map(assign_color))

# Calculate the average percentage of covered cells for each model
average_covered = df.groupby('model')['covered_cells_percentage'].mean()

# Plotting the histogram
plt.figure(figsize=(12, 8))

# Plot the bars with appropriate colors
bars = plt.bar(average_covered.index, average_covered.values, color=df.groupby('model')['color'].first(), edgecolor='black')

# Add labels and title
plt.xlabel('Model', fontsize=14)
plt.ylabel('Average % of Covered Cells', fontsize=14)
plt.title('Average Percentage of Covered Cells for Each Model', fontsize=16)

# Rotate the x-axis labels for better readability
plt.xticks(rotation=45, ha='right')

# Add legend with proper labels for the reward functions
handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10) for color in ['yellow', 'blue', 'red', 'orange']]
labels = ['Reward Function 1', 'Reward Function 2', 'Reward Function 3', 'Reward Function 4']
plt.legend(handles, labels, title="Reward Functions", loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=12)

# Show the plot
plt.tight_layout()
plt.show()

def plot_group(group_name):
    # Filter the DataFrame for the specified reward function group
    group_df = df[df['reward_function'] == group_name]
    
    # Define a color map to assign unique colors to each model
    colors = plt.cm.get_cmap('tab10', len(group_df['model'].unique()))  # Using a color map for distinct colors
    
    # Plotting the percentage of covered cells for each episode for this group
    plt.figure(figsize=(12, 8))
    
    # Loop through the models and plot their covered cells percentage for each episode
    for idx, model_name in enumerate(group_df['model'].unique()):
        model_df = group_df[group_df['model'] == model_name]
        
        # Get a color for the model
        color = colors(idx)
        
        # Plot the line for the model with a unique color
        plt.plot(model_df['episode'], model_df['covered_cells_percentage'], 
                 label=model_name, marker='o', linestyle='-', markersize=0, color=color)
        
        # Loop over each point to plot with different markers based on 'spotted' value
        for _, row in model_df.iterrows():
            marker = 'o' if row['spotted'] else 'x'  # 'o' for True (spotted), 'x' for False
            plt.scatter(row['episode'], row['covered_cells_percentage'], 
                        marker=marker, color=color)  # Scatter for different markers with the same model color
    
    # Add labels and title
    plt.xlabel('Episode', fontsize=14)
    plt.ylabel('% of Covered Cells', fontsize=14)
    plt.title(f'Percentage of Covered Cells Across Episodes for {group_name}', fontsize=16)
    
    # Move the legend outside the plot to avoid overlap
    plt.legend(title='Models', loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=12)

    # Add grid for better readability
    plt.grid(True)

    # Adjust layout to ensure everything fits without overlapping
    plt.tight_layout()

    # Show the plot
    plt.show()


# Plot for each group
plot_group('Reward Function 1')
plot_group('Reward Function 2')
plot_group('Reward Function 3')
plot_group('Reward Function 4')