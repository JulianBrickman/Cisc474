import numpy as np
import gymnasium as gym

"""
Feel free to modify the functions below and experiment with different environment configurations.
"""


def observation_space(env: gym.Env) -> gym.spaces.Space:
    """
    Observation space from Gymnasium (https://gymnasium.farama.org/api/spaces/)
    """
    return gym.spaces.Dict({
        'local_grid': gym.spaces.Box(low=0, high=6, shape=(5, 5), dtype=np.uint8),  # 5x5 grid around agent
        'agent_pos': gym.spaces.Box(low=0, high=99, shape=(2,), dtype=np.uint8),  # x,y coordinates
        'direction': gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32),  # direction to nearest uncovered cell
        'coverage': gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),  # coverage progress
        'prev_pos': gym.spaces.Box(low=0, high=99, shape=(2,), dtype=np.uint8),  # previous position
    })


def observation(grid: np.ndarray):
    """
    Function that returns the observation for the current state of the environment.
    """
    # Get agent position
    agent_pos = np.where(grid[:, :, 0] == 3)  # Find agent (grey color)
    if len(agent_pos[0]) == 0:
        agent_pos = (0, 0)  # Default position if agent not found
    else:
        agent_pos = (agent_pos[0][0], agent_pos[1][0])
    
    # Get 5x5 local grid around agent
    local_grid = np.zeros((5, 5), dtype=np.uint8)
    for i in range(-2, 3):
        for j in range(-2, 3):
            x, y = agent_pos[0] + i, agent_pos[1] + j
            if 0 <= x < 10 and 0 <= y < 10:
                # Convert RGB to cell type (0: unexplored, 1: explored, 2: wall)
                if np.array_equal(grid[x, y], [0, 0, 0]):  # Black
                    local_grid[i+2, j+2] = 0
                elif np.array_equal(grid[x, y], [255, 255, 255]):  # White
                    local_grid[i+2, j+2] = 1
                elif np.array_equal(grid[x, y], [101, 67, 33]):  # Brown
                    local_grid[i+2, j+2] = 2
    
    # Find direction to nearest uncovered cell
    direction = np.array([0.0, 0.0])
    min_dist = float('inf')
    for i in range(10):
        for j in range(10):
            if np.array_equal(grid[i, j], [0, 0, 0]):  # If cell is unexplored
                dist = np.sqrt((i - agent_pos[0])**2 + (j - agent_pos[1])**2)
                if dist < min_dist:
                    min_dist = dist
                    direction = np.array([j - agent_pos[1], i - agent_pos[0]])
                    if min_dist > 0:
                        direction = direction / min_dist
    
    # Calculate coverage progress
    total_cells = 100  # 10x10 grid
    covered_cells = np.sum(grid[:, :, 0] == 255)  # Count white cells
    coverage = np.array([covered_cells / total_cells])
    
    # Get previous position (if available)
    prev_pos = np.array([agent_pos[0], agent_pos[1]], dtype=np.uint8)
    
    return {
        'local_grid': local_grid,
        'agent_pos': np.array(agent_pos, dtype=np.uint8),
        'direction': direction,
        'coverage': coverage,
        'prev_pos': prev_pos
    }


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
    moved = info["moved"]
     # Strategy A: Exploration + Penalty for Detection
  
    if game_over and len(enemies) > 0:
        return -50

    if new_cell_covered:
        return 15  # Reward for discovering a new cell

    if moved:  # Add reward for moving
        return 0.5  # Small reward for moving
    
    # Add efficiency bonus
    if cells_remaining == 0:  # Map completed
        reward += 100  # Large bonus for completing the map

    return -0.1  # Small step penalty to encourage efficient exploration

    # Default fallback
    return 0
    # IMPORTANT: You may design a reward function that uses just some of these values. Experiment with different
    # rewards and find out what works best for the algorithm you chose given the observation space you are using

