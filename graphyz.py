import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Assuming 'evaluation_result' is the DataFrame you want to visualize
def visualize_table(df):
    # Sort the DataFrame by score in descending order
    df = df.sort_values(by='score', ascending=False)

    # Create the plot for the table visualization
    fig, ax = plt.subplots(figsize=(12, 8))  # Set a larger size for the table
    
    # Hide the axes
    ax.axis('off')

    # Create the table from the DataFrame
    table = ax.table(cellText=df.values,
                    colLabels=df.columns,
                    loc='center',
                    cellLoc='center',  # Align cells to the center
                    colColours=['#f5f5f5']*len(df.columns))  # Optional: Light color for columns

    # Customize the appearance of the table
    table.auto_set_font_size(False)  # Disable automatic font size scaling
    table.set_fontsize(10)  # Set a fixed font size
    table.scale(1.5, 1.5)  # Scale the table for better visibility

    # Color rows based on 'reward_function' column
    for i, model_name in enumerate(df['model']):
        row_color = df.loc[i, 'color']
        
        # Debug: print out the color values to check if they are valid
        print(f"Model: {model_name}, Assigned Color: {row_color}")

        # Ensure the color is a valid string (e.g., 'yellow', 'blue', etc.)
        if row_color in ['yellow', 'blue', 'red', 'orange']:
            for j in range(len(df.columns)):
                table[(i+1, j)].set_facecolor(row_color)  # Start from row 1 as row 0 is for headers
        else:
            print(f"Invalid color detected: {row_color}")

    # Show the plot
    plt.show()

# Assuming the model paths and assignment function are defined like so:
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
    return 'gray', 'gray'

# Assuming 'df' is your evaluation DataFrame

# Visualize the table

csv_path = "combined_evaluation_results.csv"  # Update with the actual path to your file

# Load the combined CSV results into a DataFrame
df = pd.read_csv(csv_path)
df['color'], df['reward_function'] = zip(*df['model'].map(assign_color))
# Create a new column for the percentage of covered cells
df['covered_cells_percentage'] = df['covered_cells'] / df['coverable_cells'] * 100
average_covered = df.groupby('model')['covered_cells_percentage'].mean()
def evaluate_models(df):
    model_evaluation = []
    
    for model_name in df['model'].unique():
        model_df = df[df['model'] == model_name].copy()
        
        # 1. Calculate the change in percentage of covered cells
        model_df.loc[:, 'covered_cells_percentage_change'] = model_df['covered_cells_percentage'].pct_change() * 100
        # Drop the NaN value from pct_change (first row will have NaN)
        model_df = model_df.dropna(subset=['covered_cells_percentage_change'])
        # 2. Calculate the average percentage of covered cells
        avg_covered_cells = model_df['covered_cells_percentage'].mean()
        
        # 3. Calculate the percentage of times spotted
        spotted_percentage = model_df['spotted'].mean() * 100
        
        # 4. Calculate the variance in the change in percentage of covered cells
        variance_covered_cells_change = model_df['covered_cells_percentage_change'].var()
        
        # 5. Lower variance is better, so we invert it for scoring purposes
        variance_score = 1 / (variance_covered_cells_change + 1e-10) 
        
        # Calculate how close the spotted percentage is to 50% (ideal)
        spotted_deviation = abs(spotted_percentage - 50)
        print(model_df['color'].iloc[0] )
        # Store the results for evaluation
        model_evaluation.append({
            'model': model_name,
            'avg_covered_cells': avg_covered_cells,
            'spotted_percentage': spotted_percentage,
            'variance_covered_cells_change': variance_covered_cells_change,
            'variance_score': variance_score,
            'spotted_deviation': spotted_deviation,
            'color': model_df['color'].iloc[0] 
        })
    
    # Convert to DataFrame for better readability
    evaluation_df = pd.DataFrame(model_evaluation)
    max_variance = evaluation_df['variance_covered_cells_change'].max()
    max_coverage = 100  # Max coverage is 100% as ideal
    
    # Normalize the metrics: higher values are better
    evaluation_df['normalized_variance'] = (evaluation_df['variance_covered_cells_change']/max_variance) * 76
    evaluation_df['spotted_score'] = 100 - (evaluation_df['spotted_deviation'] * 2)
    evaluation_df['spotted_score'] = evaluation_df['spotted_score'].clip(0, 100)
    evaluation_df['score'] = (evaluation_df['avg_covered_cells'] + 
                             evaluation_df['spotted_score'] - evaluation_df['normalized_variance'])
    
    # Sort the models by score (lower is better)
    evaluation_df = evaluation_df.sort_values(by='score', ascending=False)

    # Create table
    
#     fig, ax = plt.subplots(figsize=(10, 6))

#     # Turn off the axis
#     ax.axis('off')

#     cell_colours = [ [color] * len(df.columns) for color in df['color'] ]  # Assign the same color across each row
#     print(cell_colours)
# # Create the table, applying colors to the rows based on the 'color' column
#     table = ax.table(cellText=df.values, colLabels=df.columns, loc='center', cellColours=cell_colours)

#     # Adjust layout to ensure the table fits within the figure
#     plt.tight_layout()

#     # Show the table
#     plt.show()
    # visualize_table(evaluation_df)
    return evaluation_df

# Example usage
evaluation_dfOne= evaluate_models(df)
# Selecting the relevant columns and increasing the size of the table
evaluation_df = evaluation_dfOne[['model', 'avg_covered_cells', 'spotted_percentage', 'variance_covered_cells_change']]

# Set up the figure with larger font size
fig, ax = plt.subplots(figsize=(20, 20))

# Hide axes
ax.axis('tight')
ax.axis('off')

# Create table with larger font size and bold column headers
table = ax.table(cellText=evaluation_df.values, colLabels=evaluation_df.columns, loc='center', cellLoc='center', colColours=['lightgray']*len(evaluation_df.columns))
table.auto_set_font_size(False)
table.set_fontsize(18)

# Bold the column headers
for (i, j), cell in table.get_celld().items():
    if i == 0:
        cell.set_text_props(weight='bold')
    cell.set_fontsize(18)

# Color the rows based on the 'color' column (same logic as before)
for i, color in enumerate(evaluation_dfOne['color']):
    for j in range(len(evaluation_df.columns)):
        table[(i + 1, j)].set_facecolor(color)

# Display the table
plt.show()

