# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import random


# NOTE: This is an example strategy, you should modify it to implement your own strategy. However, DO NOT change the function signature.
def row_strategy(history):
    """
    Example function to select the next strategy for the row player based on the history of previous turns.
    Parameters:
    history (list of tuple): A list where each tuple contains the strategies used by the row and column players in previous turns.
                             Each tuple is of the form (strategy_row, strategy_col).
    Returns:
    int: The next strategy for the row player (0 or 1).
    """
    if not history: # If there's no history, start with a default strategy (e.g., choose 0)
        return 0        
    last_round = history[-1]
    rnum = random.random()
    if rnum < 0.1:
        return 0
    elif rnum < 0.2:
        return 1
    elif rnum < 0.9:
        next_strategy_row = last_round[1]    
    else: 
        next_strategy_row = last_round[0]
    return next_strategy_row
