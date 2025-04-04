import tensorflow as tf
import pandas as pd
import os

# Path to your combined results CSV
csv_path = "combined_evaluation_results.csv"

# Directory for TensorBoard logs
log_dir = "tensorboard_logs"
os.makedirs(log_dir, exist_ok=True)
writer = tf.summary.create_file_writer(log_dir)

# Load the combined CSV results into a DataFrame
df = pd.read_csv(csv_path)

# Assuming the CSV has columns: ['epoch', 'training_loss', 'training_accuracy', 'validation_loss', 'validation_accuracy']
# Iterate over the DataFrame and log each row's metrics to TensorBoard
for _, row in df.iterrows():
    epoch = int(row['episode'])  # Using 'episode' for the step
    reward = row.get('reward', None)
    steps = row.get('steps', None)
    spotted = row.get('spotted', None)
    covered_cells = row.get('covered_cells', None)
    cells_remaining = row.get('cells_remaining', None)
    steps_remaining = row.get('steps_remaining', None)
    coverable_cells = row.get('coverable_cells', None)
    model_name = row.get('model', None)
    env_name = row.get('env', None)
    with writer.as_default():
        if reward is not None:
            tf.summary.scalar(f"{model_name}_{env_name}_Reward", reward, step=epoch)
        if steps is not None:
            tf.summary.scalar(f"{model_name}_{env_name}_Steps", steps, step=epoch)
        if spotted is not None:
            tf.summary.scalar(f"{model_name}_{env_name}_Spotted", spotted, step=epoch)
        if covered_cells is not None:
            tf.summary.scalar(f"{model_name}_{env_name}_Covered_Cells", covered_cells, step=epoch)
        if cells_remaining is not None:
            tf.summary.scalar(f"{model_name}_{env_name}_Cells_Remaining", cells_remaining, step=epoch)
        if steps_remaining is not None:
            tf.summary.scalar(f"{model_name}_{env_name}_Steps_Remaining", steps_remaining, step=epoch)
        if coverable_cells is not None:
            tf.summary.scalar(f"{model_name}_{env_name}_Coverable_Cells", coverable_cells, step=epoch)
        writer.flush()
  
        

print("CSV converted to TensorBoard format at:", log_dir)
