import re

code = """
To solve this problem, we need to find the number under a given limit that produces the longest Collatz sequence. The goal is to optimize the computation to run within a one-minute runtime budget.

### Approach
The approach involves using memoization to cache the number of steps for each number encountered in the Collatz sequence. This technique significantly reduces redundant computations by storing previously calculated results, allowing us to reuse them when needed again.

1. **Memoization Cache**: We maintain a dictionary to store the number of steps required for each number to reach 1. This cache is initialized with the base case where the number 1 requires 0 steps.
2. **Iterative Calculation**: For each number from 2 up to the given limit, we compute its Collatz sequence iteratively. If the number is already in the cache, we use the cached value. If not, we compute the sequence steps and store the results in the cache.
3. **Path Tracking**: For numbers not in the cache, we track their sequence path until we encounter a number that is already cached. Once found, we compute the steps for each number in the path in reverse order and store these in the cache.

### Solution Code
```python
import time

def find_max_collatz(limit):
    max_steps = 0
    number = 0
    cache = {1: 0}

    start_time = time.time()

    for i in range(2, limit + 1):
        if i in cache:
            current_steps = cache[i]
            if current_steps > max_steps:
                max_steps = current_steps
                number = i
            continue

        path = []
        current = i

    while current not in cache:
            path.append(current)
            if current % 2 == 0:
                current = current // 2
            else:
                current = 3 * current + 1

        for num in reversed(path):
            cache[num] = cache[current] + 1
            current = num
            if num <= limit:
                if cache[num] > max_steps:
                    max_steps = cache[num]
                    number = num

    end_time = time.time()
    elapsed_time = end_time - start_time
    return (number, max_steps, elapsed_time)

limit = 10_000_000
result = find_max_collatz(limit)
print(f"The number under {limit} with the longest steps is {result[0]} (Steps: {result[1]})")
print(f"Runtime: {result[2]:.2f} seconds")
```

### Explanation
1. **Initialization**: We start by initializing a cache with the base case where the number 1 takes 0 steps to reach 1. We also initialize variables to track the maximum steps and the corresponding number.
2. **Iterating through Numbers**: For each number from 2 to the given limit, we check if it is already in the cache. If it is, we compare its step count to the current maximum and update if necessary.
3. **Tracking Collatz Path**: If the number is not in the cache, we follow its Collatz sequence, storing each encountered number in a path until we find a number that is already cached.
4. **Computing and Caching Steps**: Once a cached number is found, we compute the steps for each number in the path in reverse order, starting from the known cached number. This ensures that each number's step count is computed efficiently and stored for future use.
5. **Updating Maximum Steps**: After computing the steps for each number, we update the maximum steps and corresponding number if the newly computed steps are greater than the current maximum.

This approach efficiently reduces redundant calculations using memoization, allowing us to handle a much larger range of numbers within the given runtime constraint.
"""

def extract_code(text: str, strict=False):
    pattern = r"```(?:\s*\w+)?\n(.*?)\n```"
    matches = re.findall(pattern, text, re.DOTALL)

    if matches:
        return matches[-1]
    elif not strict:
        return text
    else:
        return ''


extracted_code = extract_code(code)

print(extracted_code)

