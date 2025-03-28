# Based on Simone Parisi's Gym Gridworlds (https://github.com/sparisi/gym_gridworlds/tree/main)
import copy
import random
import numpy as np
import gymnasium as gym
from gymnasium.error import DependencyNotInstalled
from typing import Optional
from coverage_gridworld.custom import observation_space, observation, reward

"""
THIS FILE SHOULD NOT BE MODIFIED! USE IT ONLY FOR UNDERSTANDING HOW THE ENVIRONMENT WORKS.
"""


# action IDs
LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3
STAY = 4

# rendering colors
BLACK = (0, 0, 0)            # unexplored cell
WHITE = (255, 255, 255)      # explored cell
BROWN = (101, 67, 33)        # wall
GREY = (160, 161, 161)       # agent
GREEN = (31, 198, 0)         # enemy
RED = (255, 0, 0)            # unexplored cell being observed by an enemy
LIGHT_RED = (255, 127, 127)  # explored cell being observed by an enemy

# color IDs
COLOR_IDS = {
    0: BLACK,      # unexplored cell
    1: WHITE,      # explored cell
    2: BROWN,      # wall
    3: GREY,       # agent
    4: GREEN,      # enemy
    5: RED,        # unexplored cell being observed by an enemy
    6: LIGHT_RED,  # explored cell being observed by an enemy
}


class Enemy:
    """
    Class used to manage enemy's position, orientation and the cells being observed by it (FOV Cells)
    """
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.orientation = self.random_enemy_starting_orientation()
        self.__fov_cells = []

    def __repr__(self):
        return (f"(x, y): ({self.x}, {self.y}). "
                f"Orientation: {self.__orientation_to_text()} ({self.orientation}). "
                f"FOV (x, y): {self.__fov_cells}")

    def __orientation_to_text(self):
        orientations = ["LEFT", "DOWN", "RIGHT", "UP"]
        return orientations[self.orientation]

    def rotate(self):
        self.orientation = (self.orientation + 1) % 4

    def random_enemy_starting_orientation(self):
        """
        Returns a random orientation for the enemy, but avoids having an enemy looking directly at the player,
        which would make the game fail at start
        """

        if self.y == 0:  # First row
            orientation = 1  # Starts looking Down
        elif self.x == 0:  # First column
            orientation = 0  # Starts looking Left
        else:
            orientation = random.randint(0, 3)

        return orientation

    def add_fov_cell(self, cell):
        self.__fov_cells.append(cell)

    def clear_fov_cells(self):
        self.__fov_cells = []

    def get_fov_cells(self):
        return copy.deepcopy(self.__fov_cells)


class CoverageGridworld(gym.Env):
    """
    Gridworld where the agent has to explore all tiles while avoiding enemies and obstacles.

    ## Grid
    The grid is defined by a 2D array of integers. It is possible to define custom grids.

    ## Action Space
    The action is discrete in the range `{0, 4}`.

    - 0: Move left
    - 1: Move down
    - 2: Move right
    - 3: Move up
    - 4: Stay (do not move)

    ## Observation Space
    The Observation Space must be implemented on the custom.py file. An example is already given, but we HIGHLY
    recommend that a simpler observation be used instead.

    ## Starting State
    The episode starts with the agent at the top-left tile, with that tile already explored.

    ## Transition
    The transitions are deterministic.

    ## Rewards
    The reward scheme must be implemented on the custom.py file, penalizing or rewarding certain
    behaviors (e.g. hitting a wall, not moving, walking over an explored cell, etc.). The "info" dictionary returned
    by the step method may be used for that.

    ## Episode End
    By default, an episode ends if any of the following happens:
    - The player dies (gets spotted by an enemy),
    - Explores all tiles,
    - Time runs out.

    ## Rendering
    Human mode renders the environment as a grid with colored tiles.

    - Black: unexplored tiles
    - White: explored tiles
    - Brown: walls
    - Grey: agent
    - Green: enemy
    - Red: unexplored tiles currently under enemy surveillance
    - Light red: explored tiles currently under enemy surveillance

    """

    metadata = {
        "render_modes": ["human"],
        "render_fps": 10,
        "grid_size": 10
    }

    def __init__(
            self,
            render_mode: Optional[str] = None,
            num_enemies: Optional[int] = 5,
            enemy_fov_distance: Optional[int] = 4,
            num_walls: Optional[int] = 12,
            predefined_map: Optional[np.ndarray] = None,
            predefined_map_list: Optional[list] = None,
            activate_game_status: Optional[bool] = False,
            **kwargs
    ):
        # Grid attributes
        self.grid_size = self.metadata["grid_size"]
        self.num_cells = self.grid_size * self.grid_size
        # "grid" is a numpy array which stores the RGB values of the map, being updated as the game progresses
        self.grid = np.zeros((self.grid_size, self.grid_size, 3), dtype=np.uint8)

        # Gymnasium spaces
        self.observation_space = observation_space(self)
        self.action_space = gym.spaces.Discrete(5)

        # Rendering attributes for Pygame
        self.render_mode = render_mode
        self.window_surface = None
        self.clock = None
        self.window_size = (
            min(64 * self.grid_size, 512),
            min(64 * self.grid_size, 512)
        )
        self.tile_size = (
            self.window_size[0] // self.grid_size,
            self.window_size[1] // self.grid_size,
        )

        # Map layout attributes
        self.num_enemies = num_enemies   # number of enemies used
        self.enemy_fov_distance = enemy_fov_distance   # number of cells that the enemy can observe
        self.num_walls = num_walls   # number of walls in the map

        # State attributes
        self.agent_pos = 0   # agent position, considering the flattened grid (e.g. cell 2,3 is position 23)
        self.total_covered_cells = 1   # how many cells have been covered by the agent so far
        self.coverable_cells = 0   # how many cells can be covered in the current map layout
        self.steps_remaining = 500   # steps remaining in the episode
        self.enemy_list = []   # list of enemies. Populated by __create_enemy_from_map() or __spawn_enemy_fov()
        self.game_over = False   # if the episode has ended or not

        # Environment variables
        self.predefined_map = predefined_map   # map layout definition as a list of color ids, optional
        self.activate_game_status = activate_game_status   # if game status messages should be shown or not
        self.predefined_map_list = predefined_map_list   # list of predefined maps to be used, optional
        self.current_predefined_map = 0   # index of predefined map to be used from the list

        # Validates all maps within map list to avoid program exiting after training has already begun
        self.__validate_map_list_shapes()

    def _is_color_in_cell(self, color: tuple, row: int, col: int):
        """
        Helper method to check if the value of a cell is equal to a specified color
        """
        return np.array_equal(self.grid[row, col], np.asarray(color))

    def __print_game_status(self, message):
        """
        Helper function to check if game status should be printed and to do so if needed
        """
        if self.activate_game_status:
            print(message)

    def get_state(self):
        """
        Wrapper method to return the grid
        """
        return observation(self.grid)

    def __validate_map_list_shapes(self):
        """
        Iterates through maps within map list to validate their shapes
        """
        if self.predefined_map_list is not None:
            for i, map in enumerate(self.predefined_map_list):
                if np.shape(map) != (self.grid_size, self.grid_size):
                    print(f"Invalid map dimensions for map with index {i} in list! "
                          f"Use a valid map or try random map generation.")
                    exit(1)

            self.predefined_map = self.predefined_map_list[self.current_predefined_map]

    def reset(self, **kwargs):
        """
        Required Gymnasium method, resets the environment for a new episode of training
        """
        super().reset(**kwargs)

        # Clears state variables
        self.agent_pos = 0
        self.total_covered_cells = 1
        self.steps_remaining = 500
        self.enemy_list = []
        self.game_over = False

        # Repopulates grid
        self.__populate_grid()

        # Renders map, if render_mode is "human"
        if self.render_mode is not None and self.render_mode == "human":
            self.render()

        return self.get_state(), {}

    def __populate_grid(self):
        """
        Populates grid with objects, either randomly or from a predefined map
        """
        if self.predefined_map is not None:
            # if predefined map is used, first verifies if it is valid
            self.__verify_map()
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    color_id = int(self.predefined_map[i][j])
                    self.grid[i, j] = np.asarray(COLOR_IDS[color_id])
            if not self.__is_grid_coverable():
                print("The provided map cannot be fully covered! Use a valid map or try random map generation.")
                exit(1)
            for enemy in self.enemy_list:
                self.__spawn_fov(enemy)

            # if list of maps is being used, increments counter and selects next map to be used
            if self.predefined_map_list is not None:
                self.current_predefined_map = (self.current_predefined_map + 1) % len(self.predefined_map_list)
                self.predefined_map = self.predefined_map_list[self.current_predefined_map]
        else:
            # random map generation may create invalid maps, so a limit is imposed to avoid an infinite loop
            verification_limit = 100
            for i in range(verification_limit):
                self.grid = np.zeros((self.grid_size, self.grid_size, 3), dtype=np.uint8)
                self.enemy_list = []
                self.__randomly_populate_grid()
                if self.__is_grid_coverable():
                    break
            if i == verification_limit - 1:
                print("No valid grid could be generated. Please modify environment parameters.")
                exit(1)

    def __randomly_populate_grid(self):
        """
        Assigns agent to top left corner and spawns walls, enemies and their FOV cells
        """
        occupied_cells = [0]
        self.grid[0, 0] = np.asarray(GREY)
        wall_cells = self.__spawn_items(num_items=self.num_walls, occupied_cells=occupied_cells, color=BROWN)
        occupied_cells.extend(wall_cells)
        enemy_cells = self.__spawn_items(num_items=self.num_enemies, occupied_cells=occupied_cells, color=GREEN)
        self.__spawn_enemy_fov(enemy_cells)

    def __verify_map(self):
        """
        Verifies predefined map, checking if:
        I) it has the appropriate dimensions, exiting the program if it does not
        II) if the agent is at the top left corner, correcting its position if it is not
        III) if the cells in the predefined map are either walls or enemies, ignoring any cells that are not
        """
        if np.shape(self.predefined_map) != (self.grid_size, self.grid_size):
            print("Invalid map dimensions! Use a valid map or try random map generation.")
            exit(1)

        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if i == 0 and j == 0:
                    # for the top left cell, verifies if agent was placed there
                    if self.predefined_map[i][j] != 3:
                        self.predefined_map[i][j] = 3
                elif self.predefined_map[i][j] not in [2, 4]:
                    # for other cells, only walls (color id 2) or enemies (color id 4) are accepted
                    self.predefined_map[i][j] = 0
                elif self.predefined_map[i][j] == 4:
                    # creates Enemy instances
                    self.__create_enemy_from_map(i, j)

    def __create_enemy_from_map(self, y, x):
        """
        Creates instance of Enemy from map coordinates
        """
        enemy = Enemy(x, y)
        self.enemy_list.append(enemy)

    def __spawn_items(self, num_items: int, occupied_cells: list, color: tuple):
        """
        Spawn items in random positions

        :param num_items -> number of items to be spawned
        :param occupied_cells -> cells already occupied by other items
        :param color -> color of the item to be spawned
        :return list -> list of coordinates of the objects spawned by this method
        """
        new_occupied_cells = []
        for i in range(num_items):
            random_cell_index = random.randint(1, self.num_cells - 1)
            while random_cell_index in occupied_cells or random_cell_index in new_occupied_cells:
                random_cell_index = random.randint(1, self.num_cells - 1)

            new_occupied_cells.append(random_cell_index)
            cell_row = random_cell_index // self.grid_size
            cell_col = random_cell_index % self.grid_size
            self.grid[cell_row, cell_col] = np.asarray(color)

        return new_occupied_cells

    def __spawn_enemy_fov(self, enemy_cells: list):
        """
        Creates instance of Enemy, spawns its FOV cells and adds it to enemy_list

        :param enemy_cells -> list of coordinates (x, y) where enemies should be spawned
        """
        for enemy_pos in enemy_cells:
            x = enemy_pos % self.grid_size
            y = enemy_pos // self.grid_size
            enemy = Enemy(x, y)
            self.__spawn_fov(enemy)
            self.enemy_list.append(enemy)

    def __spawn_fov(self, enemy: Enemy):
        """
        Based on the enemy's orientation and current position, spawns cells that are currently being observed by it
        """
        for i in range(1, self.enemy_fov_distance + 1):
            if enemy.orientation == 0:  # LEFT
                fov_row, fov_col = enemy.y, enemy.x - i
            elif enemy.orientation == 1:  # DOWN
                fov_row, fov_col = enemy.y + i, enemy.x
            elif enemy.orientation == 2:  # RIGHT
                fov_row, fov_col = enemy.y, enemy.x + i
            else:  # UP
                fov_row, fov_col = enemy.y - i, enemy.x

            if self.__is_cell_visible(fov_row, fov_col):
                enemy.add_fov_cell((fov_row, fov_col))
                if fov_row * self.grid_size + fov_col == self.agent_pos:
                    # if FOV cell is the agent's cell, then that creates a game over condition
                    self.game_over = True
                if self._is_color_in_cell(WHITE, fov_row, fov_col) or self._is_color_in_cell(GREY, fov_row, fov_col):
                    # if the cell was either WHITE or GREY, then it becomes LIGHT_RED
                    self.grid[fov_row, fov_col] = np.asarray(LIGHT_RED)
                elif self._is_color_in_cell(LIGHT_RED, fov_row, fov_col):
                    # if the cell was LIGHT_RED, then it becomes WHITE
                    self.grid[fov_row, fov_col] = np.asarray(WHITE)
                else:
                    # if the cell was BLACK, then it becomes RED
                    self.grid[fov_row, fov_col] = np.asarray(RED)
            else:
                # if cell is not visible, then another object is blocking it and the FOV should not be spawned
                break

    def __is_cell_visible(self, i, j):
        """
        Checks if a cell within a given coordinate is visible to an enemy that is looking in its direction
        """
        if i < 0 or j < 0 or i >= self.grid_size or j >= self.grid_size:
            # Beyond boundaries of grid
            return False
        elif self._is_color_in_cell(BROWN, i, j) or self._is_color_in_cell(GREEN, i, j):
            # Cell is already occupied by a wall or another enemy
            return False
        else:
            return True

    def __is_grid_coverable(self):
        """
        Verifies if all cells to be covered are accessible by the agent
        """
        # Boolean masks for the coverable cells in the grid
        black_cells = np.where(np.sum(self.grid, axis=2) == 0, 1, 0)
        red_cells = np.where(np.sum(self.grid, axis=2) == 255, 1, 0)
        boolean_mask = black_cells + red_cells
        # Counts number of coverable cells
        self.coverable_cells = np.sum(np.sum(boolean_mask, axis=0), axis=0) + 1

        # A simple depth-based grid walk is performed, changing the boolean masks so that coverage can be verified
        stack = {(0, 0)}
        while len(stack) > 0:
            self.__grid_walk(stack, boolean_mask)

        # Checks the number of cells that were not covered by the grid walk
        num_invalid_cells = np.count_nonzero(boolean_mask)

        return num_invalid_cells == 0

    def __grid_walk(self, stack, grid):
        """
        Depth-based grid walk to cover all unobstructed cells in the grid
        """
        head = stack.pop()
        grid[head] = 0
        neighbors = [(-1, 0), (0, -1), (1, 0), (0, 1)]

        for neighbor in neighbors:
            x = head[0] + neighbor[0]
            y = head[1] + neighbor[1]
            if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                if grid[x, y] == 1:
                    stack.add((x, y))

    def step(self, action: int):
        """
        Required Gymansium method, performs a step within the environment given the action provided
        """
        terminated = False

        if self.steps_remaining <= 0:
            return None, 0, True, False, {}

        # if action is STAY, doesn't move
        new_cell_covered = False
        if action != 4:
            new_cell_covered = self.__move(action)

        # rotates enemies after agent's movement
        self.__rotate_enemies()

        self.steps_remaining -= 1

        if self.coverable_cells == self.total_covered_cells:
            self.__print_game_status("VICTORY!")
            terminated = True
        elif self.steps_remaining <= 0:
            self.__print_game_status("TIME IS OVER!")
            terminated = True
        elif self.game_over:
            self.__print_game_status("GAME OVER!")
            terminated = True

        # creates info dictionary with extra state information
        info = {
            "enemies": self.enemy_list,
            "agent_pos": self.agent_pos,
            "total_covered_cells": self.total_covered_cells,
            "cells_remaining": self.coverable_cells - self.total_covered_cells,
            "coverable_cells": self.coverable_cells,
            "steps_remaining": self.steps_remaining,
            "new_cell_covered": new_cell_covered,
            "game_over": self.game_over
        }

        # renders the environment if needed
        if self.render_mode is not None and self.render_mode == "human":
            self.render()

        return self.get_state(), reward(info), terminated, False, info

    def __move(self, action: int):
        """
        Moves the agent within the grid based on the action provided. Returns True if a new cell is covered
        """
        movement = [(0, -1), (1, 0), (0, 1), (-1, 0)]
        agent_x = self.agent_pos % self.grid_size
        agent_y = self.agent_pos // self.grid_size
        y = agent_y + movement[action][0]
        x = agent_x + movement[action][1]
        new_cell_covered = False

        if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
            if self._is_color_in_cell(BROWN, y, x) or self._is_color_in_cell(GREEN, y, x):
                # if agent moves towards a wall or enemy, nothing happens
                pass
            else:
                self.agent_pos = y * self.grid_size + x
                # previous agent's cell becomes WHITE instead of GREY
                self.grid[agent_y, agent_x] = np.asarray(WHITE)
                # enemy rotation happens after agent movement, so RED is also accounted for
                if self._is_color_in_cell(BLACK, y, x) or self._is_color_in_cell(RED, y, x):
                    self.total_covered_cells += 1
                    new_cell_covered = True
                # new agent's cell becomes GREY
                self.grid[y, x] = np.asarray(GREY)

        return new_cell_covered

    def __rotate_enemies(self):
        """
        Iterate through enemy_list, clearing the current FOV cells and spawning new ones based on new rotation
        """
        # clears all fov cells first, to avoid race conditions
        for enemy in self.enemy_list:
            self.__clear_fov(enemy)
            enemy.rotate()

        # after all FOV cells have been cleared, new FOV cells are spawned
        for enemy in self.enemy_list:
            self.__spawn_fov(enemy)

    def __clear_fov(self, enemy):
        """
        Clears FOV cells for a given enemy
        """
        # iterates through enemy's previous FOV cells
        fov_cells = enemy.get_fov_cells()
        for cell in fov_cells:
            if self._is_color_in_cell(RED, cell[0], cell[1]):
                # a RED cell has not yet been visited by the agent, so it becomes BLACK when enemy stops observing it
                self.grid[cell] = np.asarray(BLACK)
            elif self._is_color_in_cell(LIGHT_RED, cell[0], cell[1]):
                # a LIGHT_RED cell has already been visited, so it becomes WHITE when enemy stops observing it
                self.grid[cell] = np.asarray(WHITE)
            elif (self._is_color_in_cell(BLACK, cell[0], cell[1]) or
                  self._is_color_in_cell(WHITE, cell[0], cell[1]) or
                  self._is_color_in_cell(GREY, cell[0], cell[1])):
                # cell already processed
                pass
            else:
                # error message to handle possible bugs
                print(f"---> Error! FOV cell {cell} has an invalid value: {self.grid[cell]} <---")

        # clears the list of old FOV cells
        enemy.clear_fov_cells()

    def render(self):
        """
        Renders grid to a Pygame window
        """
        try:
            import pygame
        except ImportError as e:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gymnasium[toy-text]`"
            ) from e

        if self.window_surface is None:
            pygame.init()
            pygame.display.init()
            pygame.display.set_caption(self.unwrapped.spec.id)
            self.window_surface = pygame.display.set_mode(self.window_size)

        assert (
                self.window_surface is not None
        ), "Something went wrong with pygame. This should never happen."

        if self.clock is None:
            self.clock = pygame.time.Clock()

        t_size = self.tile_size  # short notation

        # draw tiles
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                pos = (x * t_size[0], y * t_size[1])
                border = pygame.Rect(pos, tuple(cs * 1.01 for cs in t_size))
                rect = pygame.Rect(pos, tuple(cs * 0.99 for cs in t_size))

                # draw background
                if self._is_color_in_cell(WHITE, y, x):  # draws black border if cell is white
                    pygame.draw.rect(self.window_surface, BLACK, border)
                else:
                    pygame.draw.rect(self.window_surface, WHITE, border)

                if y * self.grid_size + x == self.agent_pos:  # draw agent's cell
                    if self._is_color_in_cell(GREY, y, x):  # if no enemy is observing the agent's cell
                        pygame.draw.rect(self.window_surface, WHITE, rect)
                    else:  # if an enemy is observing the agent's cell
                        pygame.draw.rect(self.window_surface, self.grid[y, x], rect)

                    agent_color = GREY
                    pygame.draw.ellipse(self.window_surface, agent_color, rect)
                else:  # draw other cells
                    pygame.draw.rect(self.window_surface, self.grid[y, x], rect)

        if self.render_mode == "human":
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:
            raise NotImplementedError

    def close(self):
        """
        Closes Pygame's window
        """
        if self.window_surface is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
