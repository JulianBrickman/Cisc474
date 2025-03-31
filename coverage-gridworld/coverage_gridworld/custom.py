import numpy as np
import gymnasium as gym

"""
Feel free to modify the functions below and experiment with different environment configurations.
"""
OBSERVATION_MODE = "window"
REWARD_FUNCTION = "rf3"
GRID_SIZE = 10

# Define colors (same as in env.py)
WHITE = np.array([255, 255, 255], dtype=np.uint8)  # explored cell
GREY = np.array([160, 161, 161], dtype=np.uint8)     # agent cell
BROWN = np.array([101, 67, 33], dtype=np.uint8)      # wall
# ------------------------------------------------------- #

# Global variable to track the last agent position (used in rf1 and rf3)
LAST_AGENT_POS = None

def observation_space(env: gym.Env) -> gym.spaces.Space:
    """
    Observation space from Gymnasium (https://gymnasium.farama.org/api/spaces/)
    """
    # The grid has (10, 10, 3) shape and can store values from 0 to 255 (uint8). To use the whole grid as the
    # observation space, we can consider a MultiDiscrete space with values in the range [0, 256).
    window_size = 9  # updated from 5 to 7
    return gym.spaces.Box(low=0.0, high=1.0, shape=(window_size, window_size, 3), dtype=np.float32)


def observation(grid: np.ndarray):
    """
    Function that returns the observation for the current state of the environment.
    """
    # If the observation returned is not the same shape as the observation_space, an error will occur!
    # Make sure to make changes to both functions accordingly.
    window_size = 9  # updated from 5 to 7
    half_window = window_size // 2  # now 3

    # Find agent position (assumes one GREY cell)
    agent_positions = np.argwhere(np.all(grid == GREY, axis=2))
    if agent_positions.shape[0] > 0:
        agent_row, agent_col = agent_positions[0]
    else:
        agent_row, agent_col = 0, 0

    # Pad the grid to allow extraction near edges
    padded_grid = np.pad(
        grid,
        ((half_window, half_window), (half_window, half_window), (0, 0)),
        mode='constant',
        constant_values=0
    )
    # Adjust agent position for padding
    agent_row_p = agent_row + half_window
    agent_col_p = agent_col + half_window

    local_view = padded_grid[
        agent_row_p - half_window: agent_row_p + half_window + 1,
        agent_col_p - half_window: agent_col_p + half_window + 1
    ]

    # Normalize RGB to [0,1]
    local_view = local_view.astype(np.float32) / 255.0
    return local_view
    # return grid.flatten()


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
    total_covered_cells = total_covered_cells = info["total_covered_cells"]
    cells_remaining = info["cells_remaining"]
    coverable_cells = info["coverable_cells"]
    steps_remaining = info["steps_remaining"]
    new_cell_covered = info["new_cell_covered"]
    game_over = info["game_over"]
     # Strategy A: Exploration + Penalty for Detection
    r = 0
    if info["new_cell_covered"]:
        r += 15 
    else:
        r-= 2
    if info["game_over"] and len(info["enemies"]) > 0:
        r-=30
    if info.get("total_covered_cells") == info.get("coverable_cells"):
        r += 500

    if info['steps_remaining'] == 0 and not (info.get("total_covered_cells") == info.get("coverable_cells")):
        r-= 100 
    r += 0.2
    return r

    # Default fallback
    return 0
    # IMPORTANT: You may design a reward function that uses just some of these values. Experiment with different
    # rewards and find out what works best for the algorithm you chose given the observation space you are using

    return 0
