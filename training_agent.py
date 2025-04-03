import os
import numpy as np
import pickle
import random
import math
import copy
import time
from student_agent import Game2048Env  # Your environment definition
import matplotlib.pyplot as plt
from collections import defaultdict
from multiprocessing import Pool

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
    def __init__(self, board_size, patterns, weights=None):
        self.board_size = board_size
        self.patterns = patterns
        if weights is not None:
            self.weights = weights  # a list of dicts for each pattern
        else:
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
        return 0 if tile == 0 else int(math.log(tile,2))
    def get_feature(self, board, coords):
        return tuple(self.tile_to_index(board[i,j]) for (i,j) in coords)
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

# --- Afterstate Simulation ---
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

# --- Splitting Condition for 15 Stages ---
# Note: 1024=2**10, 2048=2**11, 4096=2**12, 8192=2**13, 16384=2**14
def splitting_condition(stage, board):
    max_tile = np.max(board)
    def splitting_condition(stage, board):
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
    elif stage == 6:
        return (max_tile >= 4096) and (np.any(board == 2048))
    elif stage == 7:
        return (max_tile >= 4096) and (np.any(board == 2048)) and (np.any(board == 1024))
    elif stage == 8:
        return max_tile >= 8192
    elif stage == 9:
        return (max_tile >= 8192) and (np.any(board == 1024))
    elif stage == 10:
        return (max_tile >= 8192) and (np.any(board == 2048))
    elif stage == 11:
        return (max_tile >= 8192) and (np.any(board == 2048)) and (np.any(board == 1024))
    elif stage == 12:
        return (max_tile >= 8192) and (np.any(board == 4096))
    elif stage == 13:
        return (max_tile >= 8192) and (np.any(board == 4096)) and (np.any(board == 1024))
    elif stage == 14:
        return (max_tile >= 8192) and (np.any(board == 4096)) and (np.any(board == 2048))
    elif stage == 15:
        return (max_tile >= 8192) and (np.any(board == 4096)) and (np.any(board == 2048)) and (np.any(board == 1024))
    elif stage == 16:
        return max_tile >= 16384
    elif stage == 17:
        return (max_tile >= 16384) and (np.any(board == 1024))
    elif stage == 18:
        return (max_tile >= 16384) and (np.any(board == 2048))
    elif stage == 19:
        return (max_tile >= 16384) and (np.any(board == 2048)) and (np.any(board == 1024))
    elif stage == 20:
        return (max_tile >= 16384) and (np.any(board == 4096))
    elif stage == 21:
        return (max_tile >= 16384) and (np.any(board == 4096)) and (np.any(board == 1024))
    elif stage == 22:
        return (max_tile >= 16384) and (np.any(board == 4096)) and (np.any(board == 2048))
    elif stage == 23:
        return (max_tile >= 16384) and (np.any(board == 4096)) and (np.any(board == 2048)) and (np.any(board == 1024))
    elif stage == 25:
        return (max_tile >= 16384) and (np.any(board == 8192))
    elif stage == 26:
        return (max_tile >= 16384) and (np.any(board == 8192)) and (np.any(board == 1024))
    elif stage == 27:
        return (max_tile >= 16384) and (np.any(board == 8192)) and (np.any(board == 2048))
    elif stage == 28:
        return (max_tile >= 16384) and (np.any(board == 8193)) and (np.any(board == 2048)) and (np.any(board == 1024))
    elif stage == 29:
        return (max_tile >= 16384) and (np.any(board == 8192)) and (np.any(board == 4096))
    elif stage == 30:
        return (max_tile >= 16384) and (np.any(board == 8192)) and (np.any(board == 4096)) and (np.any(board == 1024))
    elif stage == 30:
        return (max_tile >= 16384) and (np.any(board == 8192)) and (np.any(board == 4096)) and (np.any(board == 2048))
    elif stage == 30:
        return (max_tile >= 16384) and (np.any(board == 8192)) and (np.any(board == 4096)) and (np.any(board == 2048)) and (np.any(board == 1024))
    elif stage == 31:
        return (max_tile >= 32768)
    else:
        return False


# --- Parallel Sample Collection Worker ---
def get_action_for_sampling(env, approximator):
    legal_moves = [a for a in range(4) if is_move_legal(env, a)]
    if not legal_moves:
        return random.choice(range(4))
    best_val = -float('inf')
    best_action = None
    for a in legal_moves:
        after, score_after, moved = get_afterstate(env, a)
        if not moved:
            continue
        val = approximator.value(after)
        if val > best_val:
            best_val = val
            best_action = a
    return best_action if best_action is not None else random.choice(legal_moves)

def collect_sample_worker(args):
    stage, approximator, start_sample = args
    env = Game2048Env()
    if start_sample is not None:
        board, score = start_sample
        env.board = board.copy()
        env.score = score
    else:
        env.reset()
    while True:
        while not env.is_game_over():
            action = get_action_for_sampling(env, approximator)
            env.step(action)
            if splitting_condition(stage, env.board):
                return (env.board.copy(), env.score)
        env.reset()

def parallel_collect_samples(stage, target_samples, num_processes, start_samples, approximator):
    pool = Pool(processes=num_processes)
    samples = []

    # Generator function that yields new arguments continuously
    def generate_args():
        while True:
            s = random.choice(start_samples) if start_samples else None
            yield (stage, approximator, s)

    args_gen = generate_args()

    # Use imap_unordered to process samples as they come in
    for result in pool.imap_unordered(collect_sample_worker, args_gen):
        samples.append(result)
        print(f"[Stage {stage}] Collected {len(samples)} samples so far...")
        if len(samples) >= target_samples:
            break

    pool.terminate()  # Stop the pool immediately
    pool.close()
    pool.join()
    return samples[:target_samples]

# --- General TD Learning Function for a Given Stage ---
def td_learning_stage(env, approximator, stage, start_samples, num_episodes, alpha, gamma, sample_limit,
                      continue_training=False, trained_flag=False, parallel_collect=False,
                      parallel_target=200, num_processes=10, early_stop_rate=0.8):
    sample_file = f"stage{stage}_samples.pkl"
    weight_file = f"stage{stage}_weights.pkl"
    
    # For Stage 1:
    if stage == 1:
        if trained_flag:
            if os.path.exists(weight_file):
                try:
                    with open(weight_file, "rb") as f:
                        saved_weights = pickle.load(f)
                    approximator.weights = saved_weights
                    if os.path.exists(sample_file):
                        with open(sample_file, "rb") as f:
                            loaded_samples = pickle.load(f)
                        print(f"[Stage 1] Loaded {len(loaded_samples)} samples and weights. Skipping training.")
                        return loaded_samples, approximator.weights
                    else:
                        print("[Stage 1] Weights found but no samples. Returning empty sample list.")
                        return [], approximator.weights
                except Exception as e:
                    print(f"[Stage 1] Error loading weights: {e}. Training from scratch.")
            else:
                print("[Stage 1] No saved weights found. Training from scratch.")
        else:
            if continue_training and os.path.exists(weight_file):
                try:
                    with open(weight_file, "rb") as f:
                        saved_weights = pickle.load(f)
                    approximator.weights = saved_weights
                    if os.path.exists(sample_file):
                        with open(sample_file, "rb") as f:
                            loaded_samples = pickle.load(f)
                        print(f"[Stage 1] Continuing training: Loaded {len(loaded_samples)} samples and weights.")
                    else:
                        loaded_samples = []
                        print("[Stage 1] No sample file found. Will generate new samples.")
                except Exception as e:
                    loaded_samples = []
                    print(f"[Stage 1] Warning: Could not load previous data: {e}. Training from scratch.")
            else:
                loaded_samples = []
        
        new_samples = []
        sample_history = []
        start_time = time.time()
        reward_history = []
        for episode in range(num_episodes):
            sample_collected = False
            env.reset()
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
                    if not moved:
                        continue
                    val = approximator.value(after)
                    if val > best_val:
                        best_val = val
                        best_action = a
                action = best_action if best_action is not None else random.choice(legal_moves)
                s_after, score_after, moved = get_afterstate(env, action)
                reward = score_after - prev_score
                prev_score = score_after
                _, _, done, _ = env.step(action)
                if (not sample_collected) and splitting_condition(stage, env.board):
                    new_samples.append((env.board.copy(), env.score))
                    sample_collected = True
                if not done:
                    legal_next = [a for a in range(4) if is_move_legal(env, a)]
                    best_val_next = -float('inf')
                    for a in legal_next:
                        next_after, next_score, moved_next = get_afterstate(env, a)
                        if not moved_next:
                            continue
                        val_next = approximator.value(next_after)
                        if val_next > best_val_next:
                            best_val_next = val_next
                    v_next = best_val_next
                else:
                    v_next = 0
                current_val = approximator.value(s_after)
                delta = reward + gamma * v_next - current_val
                approximator.update(s_after, delta, alpha)
            reward_history.append(env.score)
            sample_history.append(1 if sample_collected else 0)
            if (episode + 1) % 100 == 0:
                avg_reward = sum(reward_history[-100:]) / 100.0
                max_reward = max(reward_history[-100:])
                rate = sum(sample_history[-100:]) / 100.0
                elapsed = time.time() - start_time
                print(f"[Stage {stage}] Episode {episode+1} | Time: {elapsed:.2f}s | Avg Reward: {avg_reward:.2f} | Max Reward: {max_reward} | Splitting Rate: {rate:.2f}")
                if rate >= early_stop_rate:
                    print(f"[Stage {stage}] Early stopping: Splitting rate reached {rate:.2f} over last 100 episodes.")
                    break
        if parallel_collect:
            print(f"[Stage {stage}] Launching parallel sample collection.")
            new_samples = parallel_collect_samples(stage, parallel_target, num_processes, None if stage==1 else start_samples, approximator)
        with open(weight_file, "wb") as f:
            pickle.dump(approximator.weights, f)
        with open(sample_file, "wb") as f:
            pickle.dump(new_samples, f)
        return new_samples, approximator.weights

    # For stage > 1:
    else:
        if trained_flag:
            if os.path.exists(weight_file):
                print(f"[Stage {stage}] Loading saved weights (skip training).")
                try:
                    with open(weight_file, "rb") as f:
                        saved_weights = pickle.load(f)
                    approximator.weights = saved_weights
                    if os.path.exists(sample_file):
                        with open(sample_file, "rb") as f:
                            loaded_samples = pickle.load(f)
                        print(f"[Stage {stage}] Loaded {len(loaded_samples)} samples and weights. Skipping training.")
                        return loaded_samples, approximator.weights
                    else:
                        print(f"[Stage {stage}] Weights found but no sample file. Returning empty sample list.")
                        return [], approximator.weights
                except Exception as e:
                    print(f"[Stage {stage}] Error loading weights: {e}.")
                    return [], approximator.weights
            else:
                print(f"[Stage {stage}] No saved weight file found. Skipping training with new weights.")
                return [], approximator.weights
        if continue_training:
            if os.path.exists(weight_file):
                print(f"[Stage {stage}] Loading saved weights for continuing training...")
                try:
                    with open(weight_file, "rb") as f:
                        saved_weights = pickle.load(f)
                    approximator.weights = saved_weights
                    print(f"[Stage {stage}] Loaded weights for continuing training.")
                except Exception as e:
                    print(f"[Stage {stage}] Could not load weights: {e}. Using new weights.")
            else:
                print(f"[Stage {stage}] No saved weights found. Using new weights.")
        if not start_samples:
            print(f"[Stage {stage}] No start_samples available; breaking training loop.")
            return [], approximator.weights

        new_samples = []
        sample_history = []
        start_time = time.time()
        reward_history = []
        for episode in range(num_episodes):
            sample_collected = False
            board_sample, init_score = random.choice(start_samples)
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
                    if not moved:
                        continue
                    val = approximator.value(after)
                    if val > best_val:
                        best_val = val
                        best_action = a
                action = best_action if best_action is not None else random.choice(legal_moves)
                s_after, score_after, moved = get_afterstate(env, action)
                reward = score_after - prev_score
                prev_score = score_after
                _, _, done, _ = env.step(action)
                if (not sample_collected) and splitting_condition(stage, env.board):
                    new_samples.append((env.board.copy(), env.score))
                    sample_collected = True
                if not done:
                    legal_next = [a for a in range(4) if is_move_legal(env, a)]
                    best_val_next = -float('inf')
                    for a in legal_next:
                        next_after, next_score, moved_next = get_afterstate(env, a)
                        if not moved_next:
                            continue
                        val_next = approximator.value(next_after)
                        if val_next > best_val_next:
                            best_val_next = val_next
                    v_next = best_val_next
                else:
                    v_next = 0
                current_val = approximator.value(s_after)
                delta = reward + gamma * v_next - current_val
                approximator.update(s_after, delta, alpha)
            reward_history.append(env.score)
            sample_history.append(1 if sample_collected else 0)
            if (episode + 1) % 100 == 0:
                avg_reward = sum(reward_history[-100:]) / 100.0
                max_reward = max(reward_history[-100:])
                rate = sum(sample_history[-100:]) / 100.0
                elapsed = time.time() - start_time
                print(f"[Stage {stage}] Episode {episode+1} | Time: {elapsed:.2f}s | Avg Reward: {avg_reward:.2f} | Max Reward: {max_reward} | Splitting Rate: {rate:.2f}")
                if rate >= early_stop_rate:
                    print(f"[Stage {stage}] Early stopping: Splitting rate reached {rate:.2f} over last 100 episodes.")
                    break
        if parallel_collect:
            print(f"[Stage {stage}] Launching parallel sample collection.")
            new_samples = parallel_collect_samples(stage, parallel_target, num_processes, start_samples, approximator)
        with open(weight_file, "wb") as f:
            pickle.dump(approximator.weights, f)
        with open(sample_file, "wb") as f:
            pickle.dump(new_samples, f)
        return new_samples, approximator.weights

# --- Main Training Procedure ---
def main():
    env = Game2048Env()
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
    num_stages = 31
    episodes_per_stage = 50000
    alpha = 0.1
    gamma = 1
    sample_limit = 2000
    
    # Flags for each stage:
    # continue_flags: if True, load previous data and continue training.
    # trained_flags: if True, skip training for that stage.
    continue_flags = [False] * num_stages
    trained_flags = [False] * num_stages
    # (Set these flags as needed; for example, if you want to skip stage 1, set trained_flags[0]=True)
    continue_flags[0] = True
    trained_flags[0] = True
    
    continue_flags[1] = True
    trained_flags[1] = True
    
    continue_flags[2] = True
    trained_flags[2] = True
    
    continue_flags[3] = True
    trained_flags[3] = True
    
    continue_flags[4] = True
    trained_flags[4] = True
    
    all_weights = []
    # --- Stage 1 ---
    approximator = NTupleApproximator(board_size=4, patterns=patterns)
    print("Training Stage 1 (T1k: max>=1024)")
    stage1_samples, weights = td_learning_stage(
        env,
        approximator,
        stage=1,
        start_samples=None,
        num_episodes=episodes_per_stage,
        alpha=alpha,
        gamma=gamma,
        sample_limit=sample_limit,
        continue_training=continue_flags[0],
        trained_flag=trained_flags[0],
        parallel_collect=True,
        parallel_target=1000,
        num_processes=10,
        early_stop_rate=0.8
    )
    all_weights.append(weights)
    prev_samples = stage1_samples
    # --- Stages 2 to 15 ---
    for stage in range(2, num_stages + 1):
        approximator = NTupleApproximator(board_size=4, patterns=patterns)
        print(f"Training Stage {stage}")
        stage_samples, weights = td_learning_stage(
            env,
            approximator,
            stage=stage,
            start_samples=prev_samples,
            num_episodes=episodes_per_stage,
            alpha=alpha,
            gamma=gamma,
            sample_limit=sample_limit,
            continue_training=continue_flags[stage-1],
            trained_flag=trained_flags[stage-1],
            parallel_collect=True,
            parallel_target=1000,
            num_processes=10,
            early_stop_rate=0.8
        )
        all_weights.append(weights)
        prev_samples = stage_samples
        print(f"Stage {stage} training complete.\n")
    
    with open("all_stages_weights.pkl", "wb") as f:
        pickle.dump(all_weights, f)
    print("All stages training complete.")

if __name__ == "__main__":
    main()
