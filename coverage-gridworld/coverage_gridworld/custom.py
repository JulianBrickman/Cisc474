import numpy as np
import gymnasium as gym

"""
Feel free to modify the functions below and experiment with different environment configurations.
"""


def observation_space(env: gym.Env) -> gym.spaces.Space:
    """
    Observation space from Gymnasium (https://gymnasium.farama.org/api/spaces/)
    """
    grid_size = env.metadata["grid_size"]
    
    # Agent position (2 values: x, y)
    agent_space = gym.spaces.Box(low=0, high=grid_size-1, shape=(2,), dtype=np.int32)
    
    # Enemy information (x, y, orientation for each enemy)
    num_enemies = getattr(env, 'num_enemies', 5)
    enemy_space = gym.spaces.Box(low=0, high=grid_size-1, shape=(num_enemies * 3,), dtype=np.int32)
    
    # Local grid state (5x5 grid around agent for better context)
    local_grid_space = gym.spaces.Box(low=0, high=1, shape=(5, 5), dtype=np.int32)
    
    # Global grid state (binary: covered/uncovered)
    global_grid_space = gym.spaces.Box(low=0, high=1, shape=(grid_size, grid_size), dtype=np.int32)
    
    # Enemy view cones (binary: 1 if cell is in enemy view)
    enemy_view_space = gym.spaces.Box(low=0, high=1, shape=(grid_size, grid_size), dtype=np.int32)
    
    # Distance to nearest enemy
    nearest_enemy_space = gym.spaces.Box(low=0, high=grid_size*2, shape=(1,), dtype=np.int32)
    
    # Coverage progress (2 values: total covered, cells remaining)
    coverage_space = gym.spaces.Box(low=0, high=grid_size*grid_size, shape=(2,), dtype=np.int32)
    
    # Direction to nearest uncovered cell (2 values: dx, dy)
    direction_space = gym.spaces.Box(low=-grid_size, high=grid_size, shape=(2,), dtype=np.int32)
    
    return gym.spaces.Dict({
        'agent': agent_space,
        'enemies': enemy_space,
        'local_grid': local_grid_space,
        'global_grid': global_grid_space,
        'enemy_view': enemy_view_space,
        'nearest_enemy': nearest_enemy_space,
        'coverage': coverage_space,
        'direction': direction_space
    })


def observation(grid: np.ndarray):
    """
    Function that returns the observation for the current state of the environment.
    """
    # Extract agent position (GREY color)
    agent_pos = np.where(np.all(grid == [160, 161, 161], axis=2))
    if len(agent_pos[0]) == 0:
        agent_pos = np.array([0, 0], dtype=np.int32)
    else:
        agent_pos = np.array([agent_pos[0][0], agent_pos[1][0]], dtype=np.int32)
    
    # Extract enemy positions (GREEN color)
    enemy_positions = np.where(np.all(grid == [31, 198, 0], axis=2))
    enemy_positions = np.array([enemy_positions[0], enemy_positions[1]], dtype=np.int32).T
    
    # Pad enemy positions if less than 5 enemies
    num_enemies = 5
    if len(enemy_positions) < num_enemies:
        padding = np.zeros((num_enemies - len(enemy_positions), 2), dtype=np.int32)
        enemy_positions = np.vstack([enemy_positions, padding]).astype(np.int32)
    
    # Add dummy orientation (0) for each enemy
    enemy_orientations = np.zeros(num_enemies, dtype=np.int32)
    enemy_info = np.column_stack([enemy_positions, enemy_orientations]).astype(np.int32)
    
    # Create local 5x5 grid around agent
    local_grid = np.zeros((5, 5), dtype=np.int32)
    for i in range(-2, 3):
        for j in range(-2, 3):
            x, y = agent_pos[0] + i, agent_pos[1] + j
            if 0 <= x < grid.shape[0] and 0 <= y < grid.shape[1]:
                if np.all(grid[x, y] == [255, 255, 255]):  # WHITE color
                    local_grid[i+2, j+2] = 1
    
    # Create global binary grid state
    global_grid = np.zeros_like(grid[:,:,0], dtype=np.int32)
    covered_cells = np.where(np.all(grid == [255, 255, 255], axis=2))
    global_grid[covered_cells[0], covered_cells[1]] = 1
    
    # Create enemy view map (simplified version without actual enemy view cones)
    enemy_view = np.zeros_like(grid[:,:,0], dtype=np.int32)
    for i in range(len(enemy_positions)):
        if not np.all(enemy_positions[i] == 0):  # If not a padded enemy
            # Mark cells around enemy as potentially visible
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    x, y = enemy_positions[i][0] + dx, enemy_positions[i][1] + dy
                    if 0 <= x < grid.shape[0] and 0 <= y < grid.shape[1]:
                        enemy_view[x, y] = 1
    
    # Calculate distance to nearest enemy
    if len(enemy_positions) > 0:
        distances = np.sqrt(np.sum((enemy_positions - agent_pos)**2, axis=1))
        nearest_enemy = np.array([np.min(distances)], dtype=np.int32)
    else:
        nearest_enemy = np.array([grid.shape[0] * 2], dtype=np.int32)
    
    # Calculate coverage progress
    total_covered = np.sum(global_grid)
    cells_remaining = np.sum(global_grid == 0)
    coverage = np.array([total_covered, cells_remaining], dtype=np.int32)
    
    # Calculate direction to nearest uncovered cell
    uncovered_cells = np.where(global_grid == 0)
    if len(uncovered_cells[0]) > 0:
        # Find the closest uncovered cell
        distances = np.sqrt((uncovered_cells[0] - agent_pos[0])**2 + 
                          (uncovered_cells[1] - agent_pos[1])**2)
        closest_idx = np.argmin(distances)
        direction = np.array([
            uncovered_cells[0][closest_idx] - agent_pos[0],
            uncovered_cells[1][closest_idx] - agent_pos[1]
        ], dtype=np.int32)
    else:
        direction = np.array([0, 0], dtype=np.int32)
    
    return {
        'agent': agent_pos,
        'enemies': enemy_info.flatten(),
        'local_grid': local_grid,
        'global_grid': global_grid,
        'enemy_view': enemy_view,
        'nearest_enemy': nearest_enemy,
        'coverage': coverage,
        'direction': direction
    }


class RewardTracker:
    def __init__(self):
        self.prev_visible_enemies = 0
        self.prev_pos = (0, 0)

# Create a global tracker instance
reward_tracker = RewardTracker()

def reward(info: dict) -> float:
    """
    Function to calculate the reward for the current step based on the state information.

    The info dictionary has the following keys:
    - enemies (list): list of `Enemy` objects. Each Enemy has the following attributes:
        - x (int): column index,
        - y (int): row index,
        - orientation (int): orientation of the agent (LEFT = 0, DOWN = 1, RIGHT = 2, UP = 3),
        - fov_cells (list): list of integer tuples indicating the coordinates of cells currently observed by the agent,
    - agent_pos (int): agent position considering the flattened grid (e.g. cell `(2, 3)` corresponds to position `23`),
    - total_covered_cells (int): how many cells have been covered by the agent so far,
    - cells_remaining (int): how many cells are left to be visited in the current map layout,
    - coverable_cells (int): how many cells can be covered in the current map layout,
    - steps_remaining (int): steps remaining in the episode.
    - new_cell_covered (bool): if a cell previously uncovered was covered on this step
    - game_over (bool) : if the game was terminated because the player was seen by an enemy or not
    """
    enemies = info["enemies"]
    agent_pos = info["agent_pos"]
    total_covered_cells = info["total_covered_cells"]
    cells_remaining = info["cells_remaining"]
    coverable_cells = info["coverable_cells"]
    steps_remaining = info["steps_remaining"]
    new_cell_covered = info["new_cell_covered"]
    game_over = info["game_over"]
     # Strategy A: Exploration + Penalty for Detection
  
    if info["game_over"] and len(info["enemies"]) > 0:
        return -50

    if info["new_cell_covered"]:
        return 10  # Reward for discovering a new cell

    return -0.1  # Small step penalty to encourage efficient exploration

    # Default fallback
    return 0
    # IMPORTANT: You may design a reward function that uses just some of these values. Experiment with different
    # rewards and find out what works best for the algorithm you chose given the observation space you are using

    return 0
