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

# Global variables for reward function
REWARD_FUNCTION = "rf3"
GRID_SIZE = 10
LAST_AGENT_POS = None  # Global variable to track the last agent position
STAGNATION_COUNT = 0   # Global variable to track consecutive stagnation
STAGNATION_THRESHOLD = 4  # Threshold for how many times the agent can stay in the same position



def observation_space(env: gym.Env) -> gym.spaces.Space:
    """
    Define the observation space for the environment.
    """
    window_size = 9  # Smaller window size for better navigation
    return gym.spaces.Dict({
        'local_grid': gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(window_size, window_size, 3),
            dtype=np.float32
        ),
        'agent_pos': gym.spaces.Box(
            low=0,
            high=GRID_SIZE-1,
            shape=(2,),
            dtype=np.int32
        ),
        'direction': gym.spaces.Box(
            low=0,
            high=3,
            shape=(1,),
            dtype=np.int32
        ),
        'coverage': gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32
        ),
        'prev_pos': gym.spaces.Box(
            low=0,
            high=GRID_SIZE-1,
            shape=(2,),
            dtype=np.int32
        )
    })

def observation(grid: np.ndarray):
    """
    Function that returns the observation for the current state of the environment.
    """
    window_size = 9  # Must match observation_space
    half_window = window_size // 2

    agent_positions = np.argwhere(np.all(grid == GREY, axis=2))
    if agent_positions.shape[0] > 0:
        agent_row, agent_col = agent_positions[0]
    else:
        agent_row, agent_col = 0, 0


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


    # Normalize RGB values
    normalized_view = local_view.astype(np.float32) / 255.0

    # Calculate coverage
    total_cells = GRID_SIZE * GRID_SIZE
    covered_cells = np.sum(np.all(grid == WHITE, axis=2))
    coverage = np.array([covered_cells / total_cells], dtype=np.float32)

    # Get previous position (if available)
    global LAST_AGENT_POS
    if LAST_AGENT_POS is None:
        prev_pos = np.array([agent_row, agent_col], dtype=np.int32)
    else:
        prev_pos = np.array(LAST_AGENT_POS, dtype=np.int32)
    LAST_AGENT_POS = [agent_row, agent_col]

    # Calculate direction to nearest uncovered cell (simplified)
    direction = np.array([0], dtype=np.int32)  # Default to right

    return {
        'local_grid': normalized_view,
        'agent_pos': np.array([agent_row, agent_col], dtype=np.int32),
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
    global LAST_AGENT_POS, STAGNATION_COUNT

    enemies = info["enemies"]
    agent_pos = info["agent_pos"]
    total_covered_cells = info["total_covered_cells"]
    cells_remaining = info["cells_remaining"]
    coverable_cells = info["coverable_cells"]
    steps_remaining = info["steps_remaining"]
    new_cell_covered = info["new_cell_covered"]
    game_over = info["game_over"]
    
    # Calculate coverage multiplier (increases as fewer cells remain)
    coverage_ratio = total_covered_cells / coverable_cells
    coverage_multiplier = 1.0 + (coverage_ratio * 3.0)  # Multiplier goes from 1.0 to 3.0
    
    # Strategy A: Exploration + Penalty for Detection
    r = 0
    if info["new_cell_covered"]:
        r += 10 * coverage_multiplier  # Apply multiplier to new cell coverage reward
    if info["game_over"]:
        # Penalty decreases as more of the map is covered (encourages risk-taking near completion)
        r -= 40 * (1.0 - coverage_ratio)  # Penalty goes from -50 to 0 as coverage increases
    if info.get("total_covered_cells") == info.get("coverable_cells"):
        r += 400
    if info['steps_remaining'] == 0 and not (info.get("total_covered_cells") == info.get("coverable_cells")):
        r -= 75
    
    # Add search boost when more than 25 cells are covered
    if total_covered_cells > 60:
        r += 15 * coverage_multiplier  # Small boost that increases with coverage
    elif total_covered_cells > 40:
        r += 10 * coverage_multiplier  # Larger boost for higher coverage
    elif total_covered_cells > 25:
        r += 5 * coverage_multiplier  # Larger boost for higher coverage

    return r

    