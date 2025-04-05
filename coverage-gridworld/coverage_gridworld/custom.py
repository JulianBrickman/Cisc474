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



"""
def observation_space(env: gym.Env) -> gym.spaces.Space:
    '''
    Define the observation space as a dictionary of multiple components.
    This provides a more structured and feature-rich format for the agent.
    '''
    window_size = 9
    return gym.spaces.Dict({
        # Local grid view around the agent (9x9 window)
        'local_grid': gym.spaces.Box(
            low=0.0, 
            high=1.0, 
            shape=(window_size, window_size, 3), 
            dtype=np.float32
        ),
        # Agent's position in the grid (normalized to [0,1])
        'agent_pos': gym.spaces.Box(
            low=0.0, 
            high=1.0, 
            shape=(2,), 
            dtype=np.float32
        ),
        # Coverage statistics (percentage of map covered)
        'coverage': gym.spaces.Box(
            low=0.0, 
            high=1.0, 
            shape=(1,), 
            dtype=np.float32
        ),
        # Previous agent position (for tracking movement)
        'prev_pos': gym.spaces.Box(
            low=0.0, 
            high=1.0, 
            shape=(2,), 
            dtype=np.float32
        ),
        # Direction vector to nearest uncovered cell
        'direction': gym.spaces.Box(
            low=-1.0, 
            high=1.0, 
            shape=(2,), 
            dtype=np.float32
        )
    })


def observation(grid: np.ndarray):
    '''
    Returns a dictionary observation with multiple components.
    This provides the agent with more detailed information about the environment.
    '''
    window_size = 9
    half_window = window_size // 2

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

    # Extract local view around the agent
    local_view = padded_grid[
        agent_row_p - half_window: agent_row_p + half_window + 1,
        agent_col_p - half_window: agent_col_p + half_window + 1
    ]

    # Normalize RGB to [0,1]
    local_view = local_view.astype(np.float32) / 255.0

    # Calculate coverage statistics
    total_cells = GRID_SIZE * GRID_SIZE
    covered_cells = np.sum(np.all(grid == WHITE, axis=2))
    coverage = np.array([covered_cells / total_cells], dtype=np.float32)

    # Normalize agent position to [0,1]
    normalized_agent_pos = np.array([agent_row / (GRID_SIZE - 1), agent_col / (GRID_SIZE - 1)], dtype=np.float32)

    # Get previous position
    global LAST_AGENT_POS
    if LAST_AGENT_POS is None:
        prev_pos = normalized_agent_pos.copy()
    else:
        prev_pos = np.array([LAST_AGENT_POS[0] / (GRID_SIZE - 1), LAST_AGENT_POS[1] / (GRID_SIZE - 1)], dtype=np.float32)
    
    # Update for next time
    LAST_AGENT_POS = [agent_row, agent_col]

    # Find direction to nearest uncovered cell
    uncovered_positions = np.argwhere(~np.all(grid == WHITE, axis=2) & ~np.all(grid == BROWN, axis=2) & ~np.all(grid == GREY, axis=2))
    
    if uncovered_positions.shape[0] > 0:
        # Calculate Manhattan distances
        distances = np.abs(uncovered_positions[:, 0] - agent_row) + np.abs(uncovered_positions[:, 1] - agent_col)
        nearest_idx = np.argmin(distances)
        nearest_uncovered = uncovered_positions[nearest_idx]
        
        # Direction vector (normalized to [-1,1])
        direction_y = (nearest_uncovered[0] - agent_row) / GRID_SIZE
        direction_x = (nearest_uncovered[1] - agent_col) / GRID_SIZE
        direction = np.array([direction_y, direction_x], dtype=np.float32)
    else:
        # If no uncovered cells, point to center of grid
        center = GRID_SIZE // 2
        direction_y = (center - agent_row) / GRID_SIZE
        direction_x = (center - agent_col) / GRID_SIZE
        direction = np.array([direction_y, direction_x], dtype=np.float32)

    # Return dictionary with all observation components
    return {
        'local_grid': local_view,
        'agent_pos': normalized_agent_pos,
        'coverage': coverage,
        'prev_pos': prev_pos,
        'direction': direction
    }
"""


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
    #reward for exploring new cells
    if info["new_cell_covered"]:
        r += 15 
    else:
        #penalty for not exploring new cells
        r-= 2
    #penalty for being detected
    if info["game_over"] and len(info["enemies"]) > 0:
        r-=30
    #reward for covering all cells
    if info.get("total_covered_cells") == info.get("coverable_cells"):
        r += 500
    #penalty for not covering all cells
    if info['steps_remaining'] == 0 and not (info.get("total_covered_cells") == info.get("coverable_cells")):
        r-= 100 
    #reward for exploring new cells
    r += 0.2
    return r



"""
# ----------------------- Reward Function 1 -----------------------

def reward(info: dict) -> float:
    '''
    Reward Function 1: Basic exploration incentives with penalties for detection
    
    r_1 = +15 for any new cell covered
          +100 for completing the entire map
          -30 for being caught by enemies
          -100 for running out of steps without covering all cells
    '''
    r = 0
    
    # Reward for exploring new cells
    if info["new_cell_covered"]:
        r += 15
    
    # Reward for covering all cells
    if info["total_covered_cells"] == info["coverable_cells"]:
        r += 100
    
    # Penalty for being detected
    if info["game_over"] and len(info["enemies"]) > 0:
        r -= 30
    
    # Penalty for not covering all cells before time runs out
    if info["steps_remaining"] == 0 and info["total_covered_cells"] < info["coverable_cells"]:
        r -= 100
    
    return r
"""

"""
# ----------------------- Reward Function 2 -----------------------
# Enhanced reward scheme with stagnation penalties and coverage milestone bonuses
def reward(info: dict) -> float:
    '''
    Reward Function 2: Enhanced exploration incentives with stagnation penalties
    
    r_2 = +30 for covering a new cell
          -75 for game over
          +400 for covering all cells
          -75 for running out of steps without covering all cells
          -10 * SC for staying in the same cell (SC = stagnation count)
          
    Plus milestone bonuses:
          +15 for discovering a new cell after covering 10 cells
          +30 for discovering a new cell after covering 20 cells
          +60 for discovering a new cell after covering 30 cells
          +120 for discovering a new cell after covering 40 cells
          +240 for discovering a new cell after covering 50 cells
          +480 for discovering a new cell after covering 60 cells
          +1000 for covering all cells
          
    Plus exploration penalties:
          -300 if agent doesn't explore more than 3 cells in 10 iterations
    '''
    global LAST_AGENT_POS, STAGNATION_COUNT, POSITION_HISTORY
    
    r = 0
    
    # Extract agent position from flattened index
    row = info["agent_pos"] // GRID_SIZE
    col = info["agent_pos"] % GRID_SIZE
    current_pos = (row, col)
    
    # Track position history
    POSITION_HISTORY.append(current_pos)
    if len(POSITION_HISTORY) > 10:  # Keep only last 10 positions
        POSITION_HISTORY.pop(0)
    
    # Check stagnation (if agent is in the same position as last time)
    if LAST_AGENT_POS is not None and current_pos == tuple(LAST_AGENT_POS):
        STAGNATION_COUNT += 1
    else:
        STAGNATION_COUNT = 0
    
    # Update last position
    LAST_AGENT_POS = [row, col]
    
    # Base rewards
    if info["new_cell_covered"]:
        r += 30
    if info["game_over"] and len(info["enemies"]) > 0:
        r -= 75
    if info["total_covered_cells"] == info["coverable_cells"]:
        r += 400
    if info["steps_remaining"] == 0 and info["total_covered_cells"] < info["coverable_cells"]:
        r -= 75
    
    # Stagnation penalty
    if STAGNATION_COUNT > STAGNATION_THRESHOLD:
        r -= 10 * STAGNATION_COUNT
    
    # Milestone bonuses for new cell discoveries
    if info["new_cell_covered"]:
        total_covered = info["total_covered_cells"]
        if total_covered > 10:
            r += 15
        if total_covered > 20:
            r += 30
        if total_covered > 30:
            r += 60
        if total_covered > 40:
            r += 120
        if total_covered > 50:
            r += 240
        if total_covered > 60:
            r += 480
        if total_covered == info["coverable_cells"]:
            r += 1000
    
    # Exploration penalty - check if agent is cycling through a small number of cells
    if len(POSITION_HISTORY) >= 10:
        unique_positions = len(set(POSITION_HISTORY))
        if unique_positions <= 3:  # If agent visited 3 or fewer unique cells in the last 10 steps
            r -= 300
    
    return r
"""

"""
# ----------------------- Reward Function 3 -----------------------
# Penalty-based reward function with scaled penalties based on coverage
def reward(info: dict) -> float:
    '''
    Reward Function 3: Penalty-based system with coverage-dependent penalties
    
    r_3 = +10 for covering a new cell
          -5 for not covering a new cell
          -200 if game ends with fewer than 25 cells covered
          -100 if game ends with fewer than 35 cells covered
          -50 if game ends with fewer than 45 cells covered
          -25 if game ends with fewer than 55 cells covered
          +400 for covering all cells
    '''
    r = 0
    
    # Basic exploration reward
    if info["new_cell_covered"]:
        r += 10
    else:
        r -= 5
    
    # Completion reward
    if info["total_covered_cells"] == info["coverable_cells"]:
        r += 400
    
    # Scaled penalties based on coverage at game end
    if info["game_over"] or info["steps_remaining"] == 0:
        total_covered = info["total_covered_cells"]
        
        if total_covered < 25:
            r -= 200
        elif total_covered < 35:
            r -= 100
        elif total_covered < 45:
            r -= 50
        elif total_covered < 55:
            r -= 25
    
    return r
"""

"""
# ----------------------- Reward Function 4 -----------------------
# Dynamic reward framework with exponential scaling and logarithmic penalties
def reward(info: dict) -> float:
    '''
    Reward Function 4: Dynamic reward framework with proportional scaling
    
    r_4 = +[5 + (cells_remaining/coverable_cells)] for discovering a new cell
          -0.2 * TOTAL_REWARD if game ends unsuccessfully
          -0.2 * TOTAL_REWARD if steps run out without covering all cells
          -0.2 * TOTAL_REWARD if agent repeats fewer than 4 unique positions in N steps
          -0.2 * TOTAL_REWARD if more than 25 steps pass without covering a new cell
          -0.2 * TOTAL_REWARD if agent stays in same position beyond stagnation threshold
    '''
    global LAST_AGENT_POS, STAGNATION_COUNT, POSITION_HISTORY
    global NO_NEW_CELL_STEPS, TOTAL_REWARD
    
    r = 0
    if new_cell_covered:
        r += 1 + (0.5 * (cells_remaining / coverable_cells))
    if game_over:
        r -= (TOTAL_REWARD* 0.2)
        return r
    if info['steps_remaining'] == 0 and not (info.get("total_covered_cells") == info.get("coverable_cells")):
         r -= (TOTAL_REWARD* 0.2)
         return r
    POSITION_HISTORY.append(agent_pos)
    unique_recent_positions = len(set(POSITION_HISTORY))
    if len(POSITION_HISTORY) == POSITION_WINDOW_SIZE:
        if unique_recent_positions < 4:
            r -= TOTAL_REWARD
            TOTAL_REWARD = 0
    if STEPS_SINCE_NEW_CELL > 25:
        r -= (TOTAL_REWARD* 0.2)
    if agent_pos == LAST_AGENT_POS:
        STAGNATION_COUNT += 1
        if STAGNATION_COUNT > STAGNATION_THRESHOLD:
             r -= TOTAL_REWARD
             TOTAL_REWARD = 0
    else:
        STAGNATION_COUNT = 0 
    LAST_AGENT_POS = agent_pos

    TOTAL_REWARD += r
    return r

"""

    
