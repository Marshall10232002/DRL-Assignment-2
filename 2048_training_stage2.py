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
            self.weights = weights  # Inherit weights from previous stage
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

    # IMPROVED: Using bit_length for faster logarithm calculation
    def tile_to_index(self, tile):
        if tile == 0:
            return 0
        else:
            # math.frexp returns (mantissa, exponent) so exponent-1 gives the index.
            return math.frexp(int(tile))[1] - 1


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

# --- Splitting Condition for 11 Stages ---
def splitting_condition(stage, board):
    max_tile = np.max(board)
    if stage == 1:
        return max_tile >= 4096
    elif stage == 2:
        return max_tile >= 8192
    elif stage == 3:
        # Final stage: no splitting condition, hence no sampling.
        return False
    else:
        return False

# =============================================================================
#   TREE SEARCH CODE (TD-MCTS)
# =============================================================================

class TD_MCTS_Node:
    def __init__(self, state, score, parent=None, action=None, env=None):
        self.state = state
        self.score = score
        self.parent = parent
        self.action = action
        self.children = {}
        self.visits = 0
        self.total_reward = 0.0
        # Initialize untried actions based on legal moves.
        self.untried_actions = [a for a in range(4) if env is not None and env.is_move_legal(a)]
        
class TD_MCTS:
    def __init__(self, env, approximator, iterations=100, exploration_constant=1.41, rollout_depth=10, gamma=0.99):
        self.env = env
        self.approximator = approximator
        self.iterations = iterations
        self.c = exploration_constant
        self.rollout_depth = rollout_depth
        self.gamma = gamma

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
        for step in range(depth):
            legal_actions = [a for a in range(4) if sim_env.is_move_legal(a)]
            if not legal_actions:
                return total_reward
            best_value = -float('inf')
            best_action = None
            best_after_board = None
            best_after_score = None
            for action in legal_actions:
                after_board, after_score, _ = get_afterstate(sim_env, action)
                immediate_reward = after_score - sim_env.score
                if step == depth - 1 and not sim_env.is_game_over():
                    state_value = self.approximator.value(after_board)
                    value = immediate_reward + self.gamma * state_value
                else:
                    value = immediate_reward
                if value > best_value:
                    best_value = value
                    best_action = action
                    best_after_board = after_board
                    best_after_score = after_score
            if best_action is None:
                break
            immediate_reward = best_after_score - sim_env.score
            total_reward += discount * immediate_reward
            discount *= self.gamma
            sim_env.board = best_after_board.copy()
            sim_env.score = best_after_score
            if sim_env.is_game_over():
                return total_reward
        if not sim_env.is_game_over():
            total_reward += discount * self.approximator.value(sim_env.board)
        return total_reward

    def backpropagate(self, node, reward):
        while node is not None:
            node.visits += 1
            node.total_reward += reward
            node = node.parent

    def run_simulation(self, root):
        node = root
        sim_env = self.create_env_from_state(node.state, node.score)
        while node.untried_actions == [] and node.children:
            node = self.select_child(node)
            if node.action is not None:
                sim_env.step(node.action)
        if node.untried_actions:
            action = node.untried_actions.pop()
            sim_env.step(action)
            new_state = sim_env.board.copy()
            new_score = sim_env.score
            new_node = TD_MCTS_Node(new_state, new_score, parent=node, action=action, env=sim_env)
            new_node.untried_actions = [a for a in range(4) if sim_env.is_move_legal(a)]
            node.children[action] = new_node
            node = new_node
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

# =============================================================================
#   END TREE SEARCH CODE
# =============================================================================

# --- Modified get_action_for_sampling ---
# (This function is reused for TD learning action selection)
def get_action_for_sampling(env, approximator, use_tree_search=False, tree_search_iterations=50, 
                            exploration_constant=1.41, rollout_depth=10, gamma=0.99):
    legal_moves = [a for a in range(4) if is_move_legal(env, a)]
    if not legal_moves:
        return random.choice(range(4))
    if use_tree_search:
        root = TD_MCTS_Node(state=env.board.copy(), score=env.score, parent=None, action=None, env=env)
        root.untried_actions = legal_moves
        mcts = TD_MCTS(env, approximator, iterations=tree_search_iterations, 
                       exploration_constant=exploration_constant, rollout_depth=rollout_depth, gamma=gamma)
        for _ in range(tree_search_iterations):
            mcts.run_simulation(root)
        best_action, distribution = mcts.best_action_distribution(root)
        return best_action
    else:
        best_val = -float('inf')
        best_action = None
        pre_score = env.score
        for a in legal_moves:
            after, score_after, moved = get_afterstate(env, a)
            if not moved:
                continue
            r = score_after - pre_score
            val = r + approximator.value(after)
            if val > best_val:
                best_val = val
                best_action = a
        return best_action if best_action is not None else random.choice(legal_moves)

# --- Modified Parallel Sample Collection Worker ---
def collect_sample_worker(args):
    stage, approximator, start_sample, use_tree_search_sampling = args
    env = Game2048Env()
    if not hasattr(env, "is_move_legal"):
        env.is_move_legal = lambda a: is_move_legal(env, a)
    if start_sample is not None:
        board, score = start_sample
        env.board = board.copy()
        env.score = score
    else:
        env.reset()
    while True:
        while not env.is_game_over():
            # For sampling, we disable tree search for speed.
            action = get_action_for_sampling(env, approximator, use_tree_search=use_tree_search_sampling)
            env.step(action)
            if splitting_condition(stage, env.board):
                return (env.board.copy(), env.score)
        env.reset()

def parallel_collect_samples(stage, target_samples, num_processes, start_samples, approximator, use_tree_search_sampling):
    pool = Pool(processes=num_processes)
    samples = []
    def generate_args():
        while True:
            s = random.choice(start_samples) if start_samples else None
            yield (stage, approximator, s, use_tree_search_sampling)
    args_gen = generate_args()
    for result in pool.imap_unordered(collect_sample_worker, args_gen):
        samples.append(result)
        print(f"[Stage {stage}] Collected {len(samples)} samples so far...")
        if len(samples) >= target_samples:
            break
    pool.terminate()
    pool.close()
    pool.join()
    return samples[:target_samples]

# --- General TD Learning Function for a Given Stage ---
def td_learning_stage(env, approximator, stage, start_samples, num_episodes, alpha, gamma, sample_limit,
                      continue_training=False, trained_flag=False, parallel_collect=False,
                      parallel_target=200, num_processes=10, early_stop_rate=0.8,
                      # NEW parameters for tree search used in TD learning:
                      use_tree_search_td=True,
                      tree_search_iterations_td=10,
                      rollout_depth_td=1,
                      exploration_constant_td=1.41,
                      gamma_td=0.99,
                      # NEW parameter for sampling: disable tree search for sampling for speed.
                      use_tree_search_sampling=False):
    sample_file = f"stage{stage}_samples.pkl"
    weight_file = f"stage{stage}_weights.pkl"
    
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
                # NEW: Use tree search for action selection during TD learning.
                action = get_action_for_sampling(env, approximator, use_tree_search=use_tree_search_td,
                                                 tree_search_iterations=tree_search_iterations_td,
                                                 exploration_constant=exploration_constant_td,
                                                 rollout_depth=rollout_depth_td,
                                                 gamma=gamma_td)
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
                    prev_score = env.score
                    for a in legal_next:
                        next_after, next_score, moved_next = get_afterstate(env, a)
                        if not moved_next:
                            continue
                        next_r = next_score - prev_score
                        val_next = next_r + approximator.value(next_after)
                        if val_next > best_val_next:
                            best_val_next = val_next
                            best_next_r = next_r
                    v_next = best_val_next
                else:
                    v_next = 0
                current_val = approximator.value(s_after)
                delta = best_next_r + gamma * v_next - current_val
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
        # NEW: Save weights before launching sample collection.
        with open(weight_file, "wb") as f:
            pickle.dump(approximator.weights, f)
        if parallel_collect:
            print(f"[Stage {stage}] Launching parallel sample collection.")
            new_samples = parallel_collect_samples(stage, parallel_target, num_processes, 
                                                   None if stage==1 else start_samples, 
                                                   approximator, use_tree_search_sampling)
        with open(sample_file, "wb") as f:
            pickle.dump(new_samples, f)
        return new_samples, approximator.weights

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
                # NEW: Use tree search for TD learning action selection.
                action = get_action_for_sampling(env, approximator, use_tree_search=use_tree_search_td,
                                                 tree_search_iterations=tree_search_iterations_td,
                                                 exploration_constant=exploration_constant_td,
                                                 rollout_depth=rollout_depth_td,
                                                 gamma=gamma_td)
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
                    prev_score = env.score
                    for a in legal_next:
                        next_after, next_score, moved_next = get_afterstate(env, a)
                        if not moved_next:
                            continue
                        next_r = next_score - prev_score
                        val_next = next_r + approximator.value(next_after)
                        if val_next > best_val_next:
                            best_val_next = val_next
                            best_next_r = next_r
                    v_next = best_val_next
                else:
                    v_next = 0
                current_val = approximator.value(s_after)
                delta = best_next_r + gamma * v_next - current_val
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
        # NEW: Save weights before sample collection.
        with open(weight_file, "wb") as f:
            pickle.dump(approximator.weights, f)
        if parallel_collect and stage != 11:
            print(f"[Stage {stage}] Launching parallel sample collection.")
            new_samples = parallel_collect_samples(stage, parallel_target, num_processes, start_samples, approximator, use_tree_search_sampling)
        with open(sample_file, "wb") as f:
            pickle.dump(new_samples, f)
        return new_samples, approximator.weights

# --- Main Training Procedure ---
def main():
    env = Game2048Env()
    if not hasattr(env, "is_move_legal"):
        env.is_move_legal = lambda a: is_move_legal(env, a)
    patterns = [
        [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1)],
        [(1, 0), (1, 1), (1, 2), (1, 3), (2, 0), (2, 1)],
        [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)],
        [(1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)],
        [(0,0), (0,1),
        (1,0), (1,1)],
        [(1,0), (1,1),
        (2,0), (2,1)],
        [(1,1), (1,2),
        (2,1), (2,2)],
        [(0, 0), (1, 0), (2, 0), (3, 0)],
        [(1, 0), (1, 1), (1, 2), (1, 3)]
    ]
    num_stages = 3  # (or 11 if desired)
    episodes_per_stage = 10000
    alpha = 0.1
    gamma = 1
    sample_limit = 2000
    
    continue_flags = [False] * num_stages
    trained_flags = [False] * num_stages
    continue_flags[0] = True
    trained_flags[0] = True
    continue_flags[1] = True
    trained_flags[1] = True
    continue_flags[2] = True
    
    # NEW: Use tree search for TD learning action selection; disable for sampling.
    use_tree_search_td = False
    use_tree_search_sampling = False

    # --- Stage 1 ---
    approximator = NTupleApproximator(board_size=4, patterns=patterns)
    print("Training Stage 1 (T1k: max>=4096)")
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
        num_processes=5,
        early_stop_rate=0.9,
        use_tree_search_td=use_tree_search_td,
        tree_search_iterations_td=10,
        rollout_depth_td=1,
        exploration_constant_td=1.41,
        gamma_td=0.99,
        use_tree_search_sampling=use_tree_search_sampling
    )
    prev_samples = stage1_samples
    prev_weights = weights  # Inherit weights for the next stage

    # --- Stages 2 to num_stages ---
    for stage in range(2, num_stages + 1):
        approximator = NTupleApproximator(board_size=4, patterns=patterns, weights=prev_weights)
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
            parallel_target=10,
            num_processes=10,
            early_stop_rate=0.8,
            use_tree_search_td=use_tree_search_td,
            tree_search_iterations_td=10,
            rollout_depth_td=1,
            exploration_constant_td=1.41,
            gamma_td=0.99,
            use_tree_search_sampling=use_tree_search_sampling
        )
        prev_samples = stage_samples
        prev_weights = weights
        print(f"Stage {stage} training complete.\n")
    
    with open("final_weights.pkl", "wb") as f:
        pickle.dump(prev_weights, f)
    print("All stages training complete.")

if __name__ == "__main__":
    main()
