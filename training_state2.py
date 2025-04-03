import numpy as np
import pickle
import random
import math
import copy
import time
from student_agent import Game2048Env
import matplotlib.pyplot as plt
from collections import defaultdict

# --- Helper Function for Move Legality ---
def is_move_legal(env, action):
    board_before = env.board.copy()
    env_copy = copy.deepcopy(env)
    if action == 0:
        env_copy.move_up()
    elif action == 1:
        env_copy.move_down()
    elif action == 2:
        env_copy.move_left()
    elif action == 3:
        env_copy.move_right()
    return not np.array_equal(board_before, env_copy.board)

# --- Symmetry Transformations ---
def identity(coords, board_size=4):
    return coords
def rot90(coords, board_size=4):
    return [(j, board_size - 1 - i) for (i, j) in coords]
def rot180(coords, board_size=4):
    return [(board_size - 1 - i, board_size - 1 - j) for (i, j) in coords]
def rot270(coords, board_size=4):
    return [(board_size - 1 - j, i) for (i, j) in coords]
def reflect(coords, board_size=4):
    return [(i, board_size - 1 - j) for (i, j) in coords]

# --- NTupleApproximator ---
class NTupleApproximator:
    def __init__(self, board_size, patterns):
        self.board_size = board_size
        self.patterns = patterns
        self.weights = [defaultdict(float) for _ in patterns]
        self.symmetry_patterns = []
        for pattern in self.patterns:
            syms = self.generate_symmetries(pattern)
            self.symmetry_patterns.append(syms)
    def generate_symmetries(self, pattern):
        funcs = [lambda x: identity(x, self.board_size),
                 lambda x: rot90(x, self.board_size),
                 lambda x: rot180(x, self.board_size),
                 lambda x: rot270(x, self.board_size),
                 lambda x: reflect(x, self.board_size),
                 lambda x: rot90(reflect(x, self.board_size), self.board_size),
                 lambda x: rot180(reflect(x, self.board_size), self.board_size),
                 lambda x: rot270(reflect(x, self.board_size), self.board_size)]
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

def get_afterstate(env, action):
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

# --- TD Learning for Stage 2 ---
# In Stage 2, the splitting condition is met when the board has at least one 16384-tile and one 8192-tile.
def td_learning_stage2(env, approximator, samples, num_episodes=30000, alpha=0.1, gamma=1, sample_limit=100):
    new_samples = []  # collected samples for Stage 2
    start_time = time.time()
    reward_history = []
    for episode in range(num_episodes):
        board_sample, init_score = random.choice(samples)
        env.board = board_sample.copy()
        env.score = init_score
        prev_score = env.score
        done = False
        while not done:
            legal_moves = [a for a in range(4) if is_move_legal(env, a)]
            if not legal_moves:
                break
            best_val = -float('inf')
            best_action = None
            for a in legal_moves:
                after, score_after, moved = get_afterstate(env, a)
                if not moved: continue
                val = approximator.value(after)
                if val > best_val:
                    best_val = val
                    best_action = a
            action = best_action if best_action is not None else random.choice(legal_moves)
            s_after, score_after, moved = get_afterstate(env, action)
            reward = score_after - prev_score
            prev_score = score_after
            _, _, done, _ = env.step(action)
            if not done:
                legal_next = [a for a in range(4) if is_move_legal(env, a)]
                best_val_next = -float('inf')
                for a in legal_next:
                    next_after, next_score, moved_next = get_afterstate(env, a)
                    if not moved_next: continue
                    val_next = approximator.value(next_after)
                    if val_next > best_val_next:
                        best_val_next = val_next
                v_next = best_val_next
            else:
                v_next = 0
            current_val = approximator.value(s_after)
            delta = reward + gamma * v_next - current_val
            approximator.update(s_after, delta, alpha)
        episode_reward = env.score
        reward_history.append(episode_reward)
        if (episode + 1) % 100 == 0:
            avg_reward = sum(reward_history[-100:]) / 100.0
            max_reward = max(reward_history[-100:])
            elapsed = time.time() - start_time
            print(f"Stage 2 - Episode {episode+1} | Time: {elapsed:.2f}s | Avg Reward (last 100): {avg_reward:.2f} | Max Reward (last 100): {max_reward}")
        # Collect sample if splitting condition is met and sample limit not reached.
        if len(new_samples) < sample_limit and (np.max(env.board) >= 16384) and (np.any(env.board == 8192)):
            new_samples.append((env.board.copy(), env.score))
    return new_samples

def main():
    with open("stage1_samples.pkl", "rb") as f:
        stage1_samples = pickle.load(f)
    env = Game2048Env()
    patterns = [
        [(0,0), (1,0), (2,0), (3,0)],
        [(1,0), (1,1), (1,2), (1,3)],
        [(0,0), (0,1), (0,1), (1,1)],
        [(1,0), (1,1), (2,0), (2,1)],
        [(1,1), (1,2), (2,1), (2,2)]
    ]
    approximator = NTupleApproximator(board_size=4, patterns=patterns)
    new_samples = td_learning_stage2(env, approximator, stage1_samples, num_episodes=10000, alpha=0.1, gamma=1, sample_limit=100)
    with open("stage2_weights.pkl", "wb") as f:
        pickle.dump(approximator.weights, f)
    with open("stage2_samples.pkl", "wb") as f:
        pickle.dump(new_samples, f)
    print("Stage 2 training complete. Weights and samples saved.")

if __name__ == "__main__":
    main()
