import numpy as np
import pickle
import random
import math
import copy
import time
import matplotlib.pyplot as plt

# Import your 2048 environment and (if available) NTupleApproximator from student_agent.py
from student_agent import Game2048Env  # Assumes your environment is defined here

# ================================
# Helper functions (afterstate, splitting, etc.)
# ================================

def get_afterstate(env, action):
    """
    Given an env and an action, simulate the move (without adding a random tile)
    and return the resulting board, score, and a flag 'moved' (always True if move valid).
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

def splitting_condition(stage, board):
    """
    Check whether the board meets the splitting condition for the given stage.
    (Only stages 1 to 5 are defined here.)
    Stage 1: max tile >= 1024
    Stage 2: max tile >= 2048
    Stage 3: max tile >= 2048 and at least one 1024 exists
    Stage 4: max tile >= 4096
    Stage 5: max tile >= 4096 and at least one 1024 exists
    """
    max_tile = np.max(board)
    if stage == 1:
        return max_tile >= 1024
    elif stage == 2:
        return max_tile >= 2048
    elif stage == 3:
        return (max_tile >= 2048) and (np.any(board == 1024))
    elif stage == 4:
        return max_tile >= 4096
    elif stage == 5:
        return (max_tile >= 4096) and (np.any(board == 1024))
    else:
        return False

def get_current_stage(board, max_stage=5):
    """
    Returns the highest stage (from 1 to max_stage) whose splitting condition is met.
    If none are met, returns 1.
    """
    for s in range(max_stage, 0, -1):
        if splitting_condition(s, board):
            return s
    return 1

# ================================
# (Re)Define your NTupleApproximator if not imported.
# (This version is taken from your training code.)
# ================================
class NTupleApproximator:
    def __init__(self, board_size, patterns, weights=None):
        self.board_size = board_size
        self.patterns = patterns
        if weights is not None:
            self.weights = weights  # a list of dicts for each pattern
        else:
            from collections import defaultdict
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

# ================================
# Load saved weights (stages 1 to 5)
# ================================
with open("all_stages_weights.pkl", "rb") as f:
    all_saved_weights = pickle.load(f)  # Expecting a list [stage1_weights, stage2_weights, ..., stage5_weights]

# ================================
# Define tuple patterns (from your training code)
# ================================
patterns = [
    [(0,0), (1,0), (2,0), (3,0)],
    [(1,0), (1,1), (1,2), (1,3)],
    [(0,0), (0,1), (0,1), (1,1)],  # (Note: duplicate coordinate in pattern; adjust if needed)
    [(1,0), (1,1), (2,0), (2,1)],
    [(1,1), (1,2), (2,1), (2,2)],
    [(0,0), (0,1), (0,2), (0,3), (1,0), (1,1)],
    [(1,0), (1,1), (1,2), (1,3), (2,0), (2,1)],
    [(0,0), (0,1), (0,2), (1,0), (1,1), (1,2)],
    [(1,0), (1,1), (1,2), (2,0), (2,1), (2,2)]
]

# ================================
# Modified TD-MCTS with stage-update capability
# ================================
class TD_MCTS_Node:
    def __init__(self, state, score, parent=None, action=None):
        self.state = state          # Board state (numpy array)
        self.score = score          # Cumulative score up to this node
        self.parent = parent
        self.action = action        # Action taken from parent to reach here
        self.children = {}          # Dict: action -> child node
        self.visits = 0
        self.total_reward = 0.0
        # untried_actions will be set later using the current environment
        self.untried_actions = []

class TD_MCTS:
    def __init__(self, env, approximator, iterations=1000, exploration_constant=1.41, rollout_depth=10, gamma=0.99):
        self.env = env
        self.approximator = approximator
        self.iterations = iterations
        self.c = exploration_constant
        self.rollout_depth = rollout_depth
        self.gamma = gamma
        self.current_stage = 1  # Start with stage 1

    def create_env_from_state(self, state, score):
        new_env = copy.deepcopy(self.env)
        new_env.board = state.copy()
        new_env.score = score
        return new_env

    def select_child(self, node):
        best_score = -float('inf')
        best_child = None
        for child in node.children.values():
            if child.visits == 0:
                uct_value = float('inf')
            else:
                q_value = child.total_reward / child.visits
                uct_value = q_value + self.c * math.sqrt(math.log(node.visits) / child.visits)
            if uct_value > best_score:
                best_score = uct_value
                best_child = child
        return best_child

    def rollout(self, sim_env, depth):
        total_reward = 0.0
        discount = 1.0
        for _ in range(depth):
            legal_actions = [a for a in range(4) if sim_env.is_move_legal(a)]
            if not legal_actions:
                break  # Terminal state reached
            action = random.choice(legal_actions)
            after_board, after_score, _ = get_afterstate(sim_env, action)
            reward = after_score - sim_env.score
            total_reward += discount * reward
            discount *= self.gamma
            # Update sim_env with the afterstate
            sim_env.board = after_board
            sim_env.score = after_score
            # Check if we need to update the approximator stage during rollout
            new_stage = get_current_stage(sim_env.board)
            if new_stage > self.current_stage and new_stage <= len(all_saved_weights):
                self.current_stage = new_stage
                self.approximator.weights = all_saved_weights[new_stage - 1]
                print(f"Rollout: Updated approximator to stage {new_stage}")
        # At rollout end, if nonterminal, use the approximator to estimate value
        legal_actions = [a for a in range(4) if sim_env.is_move_legal(a)]
        if not legal_actions or sim_env.is_game_over():
            leaf_value = 0
        else:
            leaf_value = self.approximator.value(sim_env.board)
        total_reward += discount * leaf_value
        return total_reward

    def backpropagate(self, node, reward):
        while node is not None:
            node.visits += 1
            node.total_reward += reward
            node = node.parent

    def run_simulation(self, root):
        node = root
        sim_env = self.create_env_from_state(node.state, node.score)

        # Selection: traverse down the tree
        while node.untried_actions == [] and node.children:
            node = self.select_child(node)
            if node.action is not None:
                sim_env.step(node.action)
                # Update approximator stage during simulation
                new_stage = get_current_stage(sim_env.board)
                if new_stage > self.current_stage and new_stage <= len(all_saved_weights):
                    self.current_stage = new_stage
                    self.approximator.weights = all_saved_weights[new_stage - 1]
                    print(f"Simulation: Updated approximator to stage {new_stage}")

        # Expansion: if node is not terminal, expand an untried action.
        if node.untried_actions:
            action = node.untried_actions.pop()
            sim_env.step(action)
            new_state = sim_env.board.copy()
            new_score = sim_env.score
            new_node = TD_MCTS_Node(new_state, new_score, parent=node, action=action)
            # Set the new node's untried actions based on sim_env.
            new_node.untried_actions = [a for a in range(4) if sim_env.is_move_legal(a)]
            node.children[action] = new_node
            node = new_node

        # Rollout phase.
        rollout_reward = self.rollout(sim_env, self.rollout_depth)
        self.backpropagate(node, rollout_reward)

    def best_action_distribution(self, root):
        total_visits = sum(child.visits for child in root.children.values())
        distribution = np.zeros(4)
        best_visits = -1
        best_action = None
        for action, child in root.children.items():
            distribution[action] = child.visits / total_visits if total_visits > 0 else 0
            if child.visits > best_visits:
                best_visits = child.visits
                best_action = action
        return best_action, distribution

# ================================
# Main playing loop
# ================================
def main():
    # Create the game environment
    env = Game2048Env()
    env.reset()

    # Create the NTuple approximator and initialize with stage 1 weights
    approximator = NTupleApproximator(board_size=4, patterns=patterns)
    approximator.weights = all_saved_weights[0]  # Start with stage 1

    # Create TD-MCTS instance with desired parameters.
    mcts = TD_MCTS(env, approximator, iterations=500, exploration_constant=1.41, rollout_depth=10, gamma=0.99)

    # Play until game over.
    while not env.is_game_over():
        # Create the root node from current env state.
        root = TD_MCTS_Node(env.board.copy(), env.score, parent=None, action=None)
        # Set legal moves at root.
        root.untried_actions = [a for a in range(4) if env.is_move_legal(a)]

        # Run MCTS simulations starting from the root.
        for _ in range(mcts.iterations):
            mcts.run_simulation(root)

        best_action, distribution = mcts.best_action_distribution(root)
        print(f"Chosen action: {best_action} with visit distribution: {distribution}")

        # Apply the chosen action in the real environment.
        env.step(best_action)
        env.render()
        time.sleep(0.5)  # Pause a bit to observe the game

        # Check and update approximator stage if needed.
        new_stage = get_current_stage(env.board)
        if new_stage > mcts.current_stage and new_stage <= len(all_saved_weights):
            mcts.current_stage = new_stage
            approximator.weights = all_saved_weights[new_stage - 1]
            print(f"Main loop: Updated approximator to stage {new_stage}")

    print("Game Over!")
    print("Final Score:", env.score)
    plt.show()  # In case render() uses matplotlib

if __name__ == '__main__':
    main()
