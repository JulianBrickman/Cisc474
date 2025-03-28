from gymnasium.envs.registration import register
from coverage_gridworld.env import CoverageGridworld

register(
    id="standard",
    entry_point="coverage_gridworld:CoverageGridworld"
)

register(
    id="just_go",   # very easy difficulty
    entry_point="coverage_gridworld:CoverageGridworld",
    kwargs={
        "predefined_map": [
            [3, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        ]
    }
)

register(
    id="safe",   # easy difficulty
    entry_point="coverage_gridworld:CoverageGridworld",
    kwargs={
        "predefined_map": [
            [3, 0, 0, 2, 0, 2, 0, 0, 0, 0],
            [0, 2, 0, 0, 0, 2, 0, 0, 2, 0],
            [0, 2, 0, 2, 2, 2, 2, 2, 2, 0],
            [0, 2, 0, 0, 0, 2, 0, 0, 0, 0],
            [0, 2, 0, 2, 0, 2, 0, 0, 2, 0],
            [0, 2, 0, 0, 0, 0, 0, 2, 0, 0],
            [0, 2, 2, 2, 0, 0, 0, 2, 0, 0],
            [0, 0, 0, 0, 0, 2, 0, 0, 2, 0],
            [0, 2, 0, 2, 0, 2, 2, 0, 0, 0],
            [0, 0, 0, 0, 0, 2, 0, 0, 0, 0]
        ]
    }
)

register(
    id="maze",   # medium difficulty
    entry_point="coverage_gridworld:CoverageGridworld",
    kwargs={
        "predefined_map": [
            [3, 2, 0, 0, 0, 0, 2, 0, 0, 0],
            [0, 2, 0, 2, 2, 0, 2, 0, 2, 2],
            [0, 2, 0, 2, 0, 0, 2, 0, 0, 0],
            [0, 2, 0, 2, 0, 2, 2, 2, 2, 0],
            [0, 2, 0, 2, 0, 0, 2, 0, 0, 0],
            [0, 2, 0, 2, 2, 0, 2, 0, 2, 2],
            [0, 2, 0, 2, 0, 0, 2, 0, 0, 0],
            [0, 2, 0, 2, 0, 2, 2, 2, 2, 0],
            [0, 2, 0, 2, 0, 4, 2, 4, 0, 0],
            [0, 0, 0, 2, 0, 0, 0, 0, 0, 0]
        ]
    }
)

register(
    id="chokepoint",   # hard difficulty
    entry_point="coverage_gridworld:CoverageGridworld",
    kwargs={
        "predefined_map": [
            [3, 0, 2, 0, 0, 0, 0, 2, 0, 0],
            [0, 0, 2, 0, 0, 0, 0, 0, 0, 4],
            [0, 0, 2, 0, 0, 0, 0, 2, 0, 0],
            [0, 0, 2, 0, 0, 0, 0, 2, 0, 0],
            [0, 4, 2, 0, 0, 0, 0, 2, 0, 0],
            [0, 0, 2, 0, 0, 0, 0, 2, 0, 0],
            [0, 0, 2, 0, 0, 0, 0, 2, 0, 0],
            [0, 0, 2, 0, 0, 0, 0, 2, 0, 0],
            [0, 0, 0, 0, 4, 0, 4, 2, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 2, 0, 0]
        ]
    }
)

register(
    id="sneaky_enemies",   # very hard difficulty
    entry_point="coverage_gridworld:CoverageGridworld",
    kwargs={
        "predefined_map": [
            [3, 0, 0, 0, 0, 0, 0, 4, 0, 0],
            [0, 2, 0, 2, 0, 0, 2, 0, 2, 0],
            [0, 0, 0, 0, 4, 0, 0, 0, 0, 0],
            [0, 2, 0, 2, 0, 0, 2, 0, 2, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 2, 0, 2, 0, 0, 2, 0, 2, 0],
            [4, 0, 0, 0, 0, 0, 0, 0, 0, 4],
            [0, 2, 0, 2, 0, 0, 2, 0, 2, 0],
            [0, 0, 0, 0, 0, 4, 0, 0, 0, 0],
            [0, 2, 0, 2, 0, 0, 2, 0, 2, 0]
        ]
    }
)

# To create a predefined map, just add walls and enemies. The agent always starts in the top-left corner.
# The enemy's orientation is randomly defined and the cells under surveillance will be spawned automatically
