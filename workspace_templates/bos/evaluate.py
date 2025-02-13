import json

import numpy as np
from strategy import row_strategy
from target import target_column_strategy


def get_bos_payoff_matrix():
    """Payoff matrix for the Battle of the Sexes game."""
    bos_payoff_matrix = [
        [(2, 1), (0, 0)],  # Row player chooses 0
        [(0, 0), (1, 2)]   # Row player chooses 1
    ]
    return bos_payoff_matrix

def game_payoffs(payoff_matrix, strategy_row: int, strategy_col: int):
    """Get game payoff for given strategies (tuple)."""
    if strategy_row not in [0, 1] or strategy_col not in [0, 1]:
        raise ValueError("Strategies must be either 0 or 1")
    return payoff_matrix[strategy_row][strategy_col]

def run_iterated_game(rounds: int, row_strategy_func, col_strategy_func):
    """Run iterated contest for given number of rounds with specified iterated strategy functions."""
    history = []    
    for _ in range(rounds):
        next_move_row = row_strategy_func(history)
        next_move_col = col_strategy_func(history)
        history.append((next_move_row, next_move_col))
    return history

def compute_overall_scores(payoff_matrix, history):
    """Compute final scores for both sides."""
    result_game_payoffs = [game_payoffs(payoff_matrix, strat_profile[0], strat_profile[1]) for strat_profile in history]
    row_payoffs = [payoff_tupp[0] for payoff_tupp in result_game_payoffs]
    col_payoffs = [payoff_tupp[1] for payoff_tupp in result_game_payoffs]
    np_row_payoffs = np.asarray(row_payoffs, dtype=np.float32)
    np_col_payoffs = np.asarray(col_payoffs, dtype=np.float32)
    return np.mean(np_row_payoffs), np.mean(np_col_payoffs)

def simulation_mean_scores(payoff_matrix, row_strategy_func, col_strategy_func, num_rounds, num_simulations):
    """Run Monte-Carlo simulation, return mean score estimates."""
    row_scores = []
    col_scores = []
    for _ in range(num_simulations):
        final_history = run_iterated_game(num_rounds, row_strategy_func, col_strategy_func)
        row_score, col_score = compute_overall_scores(payoff_matrix, final_history)
        row_scores.append(row_score)
        col_scores.append(col_score)
    np_row_scores = np.asarray(row_scores, dtype=np.float32)
    np_col_scores = np.asarray(col_scores, dtype=np.float32)
    row_mean = np.mean(np_row_scores)
    col_mean = np.mean(np_col_scores)
    return row_mean, col_mean

def evaluate():
    row_score, _ = simulation_mean_scores(get_bos_payoff_matrix(), row_strategy, target_column_strategy, 10, 10000)
    metrics = {"score": float(row_score)}
    print(json.dumps(metrics))
    
evaluate()
