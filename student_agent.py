import copy
import random
import math
import numpy as np
import pickle
import time
import matplotlib.pyplot as plt
from collections import defaultdict
import gym
from gym import spaces
import gc  # for memory cleanup

# ------------------------------
# Game2048 Environment
# ------------------------------

class Game2048Env(gym.Env):
    def __init__(self):
        super(Game2048Env, self).__init__()
        self.size = 4  # 4x4 2048 board
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.score = 0
        # Action space: 0: up, 1: down, 2: left, 3: right
        self.action_space = spaces.Discrete(4)
        self.actions = ["up", "down", "left", "right"]
        self.last_move_valid = True  # Record if the last move was valid
        self.reset()

    def reset(self):
        """Reset the environment"""
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.score = 0
        self.add_random_tile()
        self.add_random_tile()
        return self.board

    def add_random_tile(self):
        """Add a random tile (2 or 4) to an empty cell"""
        empty_cells = list(zip(*np.where(self.board == 0)))
        if empty_cells:
            x, y = random.choice(empty_cells)
            self.board[x, y] = 2 if random.random() < 0.9 else 4

    def compress(self, row):
        """Compress the row: move non-zero values to the left"""
        new_row = row[row != 0]  # Remove zeros
        new_row = np.pad(new_row, (0, self.size - len(new_row)), mode='constant')
        return new_row

    def merge(self, row):
        """Merge adjacent equal numbers in the row"""
        for i in range(len(row) - 1):
            if row[i] == row[i + 1] and row[i] != 0:
                row[i] *= 2
                row[i + 1] = 0
                self.score += row[i]
        return row

    def move_left(self):
        """Move the board left"""
        moved = False
        for i in range(self.size):
            original_row = self.board[i].copy()
            new_row = self.compress(self.board[i])
            new_row = self.merge(new_row)
            new_row = self.compress(new_row)
            self.board[i] = new_row
            if not np.array_equal(original_row, self.board[i]):
                moved = True
        return moved

    def move_right(self):
        """Move the board right"""
        moved = False
        for i in range(self.size):
            original_row = self.board[i].copy()
            reversed_row = self.board[i][::-1]
            reversed_row = self.compress(reversed_row)
            reversed_row = self.merge(reversed_row)
            reversed_row = self.compress(reversed_row)
            self.board[i] = reversed_row[::-1]
            if not np.array_equal(original_row, self.board[i]):
                moved = True
        return moved

    def move_up(self):
        """Move the board up"""
        moved = False
        for j in range(self.size):
            original_col = self.board[:, j].copy()
            col = self.compress(self.board[:, j])
            col = self.merge(col)
            col = self.compress(col)
            self.board[:, j] = col
            if not np.array_equal(original_col, self.board[:, j]):
                moved = True
        return moved

    def move_down(self):
        """Move the board down"""
        moved = False
        for j in range(self.size):
            original_col = self.board[:, j].copy()
            reversed_col = self.board[:, j][::-1]
            reversed_col = self.compress(reversed_col)
            reversed_col = self.merge(reversed_col)
            reversed_col = self.compress(reversed_col)
            self.board[:, j] = reversed_col[::-1]
            if not np.array_equal(original_col, self.board[:, j]):
                moved = True
        return moved

    def is_game_over(self):
        """Check if there are no legal moves left"""
        if np.any(self.board == 0):
            return False
        for i in range(self.size):
            for j in range(self.size - 1):
                if self.board[i, j] == self.board[i, j+1]:
                    return False
        for j in range(self.size):
            for i in range(self.size - 1):
                if self.board[i, j] == self.board[i+1, j]:
                    return False
        return True

    def step(self, action):
        """Execute one action"""
        assert self.action_space.contains(action), "Invalid action"
        if action == 0:
            moved = self.move_up()
        elif action == 1:
            moved = self.move_down()
        elif action == 2:
            moved = self.move_left()
        elif action == 3:
            moved = self.move_right()
        else:
            moved = False
        self.last_move_valid = moved
        if moved:
            self.add_random_tile()
        done = self.is_game_over()
        return self.board, self.score, done, {}

    def render(self, mode="human", action=None):
        """Render the current board using Matplotlib"""
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(-0.5, self.size - 0.5)
        ax.set_ylim(-0.5, self.size - 0.5)
        COLOR_MAP = {0:"#cdc1b4", 2:"#eee4da", 4:"#ede0c8", 8:"#f2b179", 16:"#f59563",
                     32:"#f67c5f", 64:"#f65e3b", 128:"#edcf72", 256:"#edcc61", 512:"#edc850",
                     1024:"#edc53f", 2048:"#edc22e"}
        TEXT_COLOR = {2:"black", 4:"black"}
        for i in range(self.size):
            for j in range(self.size):
                value = self.board[i, j]
                color = COLOR_MAP.get(value, "#3c3a32")
                text_color = TEXT_COLOR.get(value, "white")
                rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=color, edgecolor="black")
                ax.add_patch(rect)
                if value != 0:
                    ax.text(j, i, str(value), ha='center', va='center',
                            fontsize=16, fontweight='bold', color=text_color)
        title = f"score: {self.score}"
        if action is not None:
            title += f" | action: {self.actions[action]}"
        plt.title(title)
        plt.gca().invert_yaxis()
        plt.show()

    def simulate_row_move(self, row):
        """Simulate a left move for a single row"""
        new_row = row[row != 0]
        new_row = np.pad(new_row, (0, self.size - len(new_row)), mode='constant')
        for i in range(len(new_row) - 1):
            if new_row[i] == new_row[i + 1] and new_row[i] != 0:
                new_row[i] *= 2
                new_row[i + 1] = 0
        new_row = new_row[new_row != 0]
        new_row = np.pad(new_row, (0, self.size - len(new_row)), mode='constant')
        return new_row

    def is_move_legal(self, action):
        """Check if the specified move is legal (i.e., changes the board)"""
        temp_board = self.board.copy()
        if action == 0:
            for j in range(self.size):
                col = temp_board[:, j]
                new_col = self.simulate_row_move(col)
                temp_board[:, j] = new_col
        elif action == 1:
            for j in range(self.size):
                col = temp_board[:, j][::-1]
                new_col = self.simulate_row_move(col)
                temp_board[:, j] = new_col[::-1]
        elif action == 2:
            for i in range(self.size):
                row = temp_board[i]
                temp_board[i] = self.simulate_row_move(row)
        elif action == 3:
            for i in range(self.size):
                row = temp_board[i][::-1]
                new_row = self.simulate_row_move(row)
                temp_board[i] = new_row[::-1]
        else:
            raise ValueError("Invalid action")
        return not np.array_equal(self.board, temp_board)

# ------------------------------
# TD & Afterstate Functions and Approximator Setup
# ------------------------------

def get_afterstate(env, action):
    """
    Simulate the move (without adding a random tile) and return the afterstate.
    """
    env_copy = copy.deepcopy(env)
    if action == 0:
        env_copy.move_up()
    elif action == 1:
        env_copy.move_down()
    elif action == 2:
        env_copy.move_left()
    elif action == 3:
        env_copy.move_right()
    return env_copy.board.copy(), env_copy.score, True

def get_current_stage(board, max_stage=100):
    """
    Returns the current stage based on the board's maximum tile and the presence of a 1024 tile.
    Stages are defined as:
      Stage 1: max tile < 1024.
      Stage 2: 1024 ≤ max tile < 2048.
      Stage 3: 2048 ≤ max tile < 4096 and NO 1024 tile present.
      Stage 4: 2048 ≤ max tile < 4096 and a 1024 tile is present.
      Stage 5: max tile ≥ 4096.
    """
    max_tile = np.max(board)
    if max_tile < 1024:
        return 1
    elif max_tile < 2048:
        return 2
    elif max_tile < 4096:
        if np.any(board == 1024):
            return 4
        else:
            return 3
    elif max_tile < 8192:
        if np.any(board == 2048):
            if np.any(board == 1024):
                return 8
            else:
                return 7
        elif np.any(board == 1024):
            return 6
        else:
            return 5
    else:
        return 8

# --- NTupleApproximator ---
class NTupleApproximator:
    def __init__(self, board_size, patterns, weights=None):
        self.board_size = board_size
        self.patterns = patterns
        if weights is not None:
            self.weights = weights  # a list of dicts for each pattern
        else:
            self.weights = [defaultdict(float) for _ in patterns]

        # Precompute symmetry transformations for each pattern.
        self.symmetry_patterns = []
        for pattern in self.patterns:
            syms = self.generate_symmetries(pattern)
            self.symmetry_patterns.append(syms)

    def generate_symmetries(self, pattern):
        def identity(coords, board_size):
            return coords
        def rot90(coords, board_size):
            return [(j, board_size - 1 - i) for (i, j) in coords]
        def rot180(coords, board_size):
            return [(board_size - 1 - i, board_size - 1 - j) for (i, j) in coords]
        def rot270(coords, board_size):
            return [(board_size - 1 - j, i) for (i, j) in coords]
        def reflect(coords, board_size):
            return [(i, board_size - 1 - j) for (i, j) in coords]

        funcs = [
            lambda x: identity(x, self.board_size),
            lambda x: rot90(x, self.board_size),
            lambda x: rot180(x, self.board_size),
            lambda x: rot270(x, self.board_size),
            lambda x: reflect(x, self.board_size),
            lambda x: rot90(reflect(x, self.board_size), self.board_size),
            lambda x: rot180(reflect(x, self.board_size), self.board_size),
            lambda x: rot270(reflect(x, self.board_size), self.board_size)
        ]
        syms = []
        for f in funcs:
            transformed = f(pattern)
            if transformed not in syms:
                syms.append(transformed)
        return syms

    def tile_to_index(self, tile):
        return 0 if tile == 0 else int(math.log(tile, 2))

    def get_feature(self, board, coords):
        return tuple(self.tile_to_index(board[i, j]) for (i, j) in coords)

    def value(self, board):
        total = 0.0
        for weight_dict, sym_group in zip(self.weights, self.symmetry_patterns):
            group_val = 0.0
            for pattern in sym_group:
                feature = self.get_feature(board, pattern)
                group_val += weight_dict[feature]
            total += group_val / len(sym_group)
        return total

    def update(self, board, delta, alpha):
        for weight_dict, sym_group in zip(self.weights, self.symmetry_patterns):
            for pattern in sym_group:
                feature = self.get_feature(board, pattern)
                weight_dict[feature] += (alpha * delta) / len(sym_group)

# Define your n-tuple patterns
patterns = [
    [(0,0), (1,0), (2,0), (3,0)],
    [(1,0), (1,1), (1,2), (1,3)],
    [(0,0), (0,1), (0,1), (1,1)],
    [(1,0), (1,1), (2,0), (2,1)],
    [(1,1), (1,2), (2,1), (2,2)],
    [(0,0), (0,1), (0,2), (0,3), (1,0), (1,1)],
    [(1,0), (1,1), (1,2), (1,3), (2,0), (2,1)],
    [(0,0), (0,1), (0,2), (1,0), (1,1), (1,2)],
    [(1,0), (1,1), (1,2), (2,0), (2,1), (2,2)]
]

# Global approximator variable
approximator = None

# Load stage weights for stages 1–8
all_saved_weights = []  # all_saved_weights[0] corresponds to stage 1, etc.
num_stages = 8
for stage in range(1, num_stages + 1):
    filename = f"stage{stage}_weights.pkl"
    try:
        with open(filename, "rb") as f:
            weights = pickle.load(f)
        all_saved_weights.append(weights)
        print(f"Loaded weights for stage {stage} from {filename}.")
    except FileNotFoundError:
        print(f"Could not find {filename}. Using empty weights for stage {stage}.")
        empty_weights = [defaultdict(float) for _ in patterns]
        all_saved_weights.append(empty_weights)

# ------------------------------
# Global Model Initialization (TA Suggestion)
# ------------------------------

def init_model():
    """
    Initialize the global approximator model. Uses gc.collect() for memory cleanup.
    """
    global approximator
    if approximator is None:
        gc.collect() 
        approximator = NTupleApproximator(board_size=4, patterns=patterns)
        # Initialize with stage 1 weights
        approximator.weights = all_saved_weights[0]

# ------------------------------
# TD & Afterstate Simulation Functions
# ------------------------------

def rollout_td(sim_env, approximator_local, rollout_depth=5, gamma=0.99):
    """
    Rollout from the current simulation environment using a TD (greedy) policy.
    At each step, for each legal action, compute its afterstate to obtain the immediate reward and the approximator's estimated value.
    If the afterstate indicates a stage transition, use the corresponding weights.
    """
    total_rollout = 0.0
    discount = 1.0
    depth = 0
    current_stage = get_current_stage(sim_env.board, max_stage=100)
    
    while depth < rollout_depth and not sim_env.is_game_over():
        legal_actions = [a for a in range(4) if sim_env.is_move_legal(a)]
        if not legal_actions:
            break
        
        best_action = None
        best_estimate = -float('inf')
        
        for action in legal_actions:
            board_after, score_after, _ = get_afterstate(sim_env, action)
            immediate_reward = score_after - sim_env.score
            new_stage = get_current_stage(board_after, max_stage=100)
            if new_stage != current_stage:
                temp_approx = NTupleApproximator(sim_env.board.shape[0], patterns, weights=all_saved_weights[new_stage - 1])
                value_est = immediate_reward + gamma * temp_approx.value(board_after)
            else:
                value_est = immediate_reward + gamma * approximator_local.value(board_after)
            if value_est > best_estimate:
                best_estimate = value_est
                best_action = action
        
        prev_score = sim_env.score
        sim_env.step(best_action)
        r = sim_env.score - prev_score
        total_rollout += discount * r
        discount *= gamma
        depth += 1
        current_stage = get_current_stage(sim_env.board, max_stage=100)
    
    if not sim_env.is_game_over():
        total_rollout += discount * approximator_local.value(sim_env.board)
    return total_rollout

def simulate_action(action, env, approximator_local, rollout_depth=5, gamma=0.99, num_simulations=20):
    """
    For a given action from the current state, simulate multiple playouts using rollout_td.
    Return the average return (immediate reward plus rollout reward).
    """
    rewards = []
    for _ in range(num_simulations):
        sim_env = copy.deepcopy(env)
        prev_score = sim_env.score
        sim_env.step(action)
        immediate_reward = sim_env.score - prev_score
        rollout_reward = rollout_td(sim_env, approximator_local, rollout_depth, gamma)
        rewards.append(immediate_reward + rollout_reward)
    return np.mean(rewards)

# ------------------------------
# get_action Function (Called by eval.py)
# ------------------------------

def get_action(state, score):
    """
    Given the current state (board) and score, select an action using TD-MCTS simulation.
    This function creates a temporary environment, sets its board and score,
    and uses the global approximator model.
    """
    global approximator
    # Ensure the global model is initialized (also cleans memory if needed)
    if approximator is None:
        init_model()
    
    # Perform memory cleanup only at the start of the game: score is 0 and only 2 tiles are present.
    if score == 0 and np.count_nonzero(state) == 2:
        gc.collect()
    
    # Create a temporary environment with the given state and score.
    env = Game2048Env()
    env.board = np.copy(state)
    env.score = score
    print(env.board)
    
    # Use the global approximator (assumed to be stage 1 weights initially)
    approximator_local = approximator
    
    legal_actions = [a for a in range(4) if env.is_move_legal(a)]
    if not legal_actions:
        return random.choice([0, 1, 2, 3])
    
    action_values = {}
    for action in legal_actions:
        value_estimate = simulate_action(
            action, env, approximator_local, rollout_depth=2, gamma=1, num_simulations=5
        )
        action_values[action] = value_estimate
    best_action = max(action_values, key=action_values.get)
    return best_action


# ------------------------------
# Main Function for Testing (Not used in eval.py)
# ------------------------------
def main():
    init_model()  # Initialize the global approximator once per game
    env = Game2048Env()
    env.reset() 
    while not env.is_game_over():
        action = get_action(env.board, env.score)
        env.step(action)
        print(env.board)
    print("Final score:", env.score)
    
if __name__ == '__main__':
    main()