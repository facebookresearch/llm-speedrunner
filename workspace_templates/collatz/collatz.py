# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import time

def collatz_steps(n):
    steps = 0
    while n != 1:
        if n % 2 == 0:
            n //= 2
        else:
            n = 3 * n + 1
        steps += 1
    return steps

def find_max_collatz(limit):
    max_steps = 0
    number = 0
    start_time = time.time()  # Start timing
    
    for i in range(1, limit + 1):
        steps = collatz_steps(i)
        if steps > max_steps:
            max_steps = steps
            number = i

    end_time = time.time()  # End timing
    elapsed_time = end_time - start_time
    return number, max_steps, elapsed_time

limit = 10_000_000
result = find_max_collatz(limit)
print(f"limit: {limit} start_value: {result[0]} max_steps: {result[1]} runtime: {result[2]:.2f}")