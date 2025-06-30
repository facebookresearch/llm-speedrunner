# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional
import json
import os
import re

import matplotlib.pyplot as plt
import math
import numpy as np
import pandas as pd


def gather_metrics(
    workspace_path: str, 
    metrics: list[str],
    workspace_template_path: Optional[str] = None
) -> pd.DataFrame:
    data = {'step': []}
    for m in metrics:
        data[m] = []

    step2path = {}

    if workspace_template_path is not None:
        initial_result_path = os.path.join(workspace_template_path, 'results.json')
        if os.path.exists(initial_result_path):
            step2path[0] = initial_result_path

    # Gather metrics from version subdirectories
    version_dir_pattern = re.compile(r"^v_(\d+)$")
    for entry in os.scandir(workspace_path):
        if entry.is_dir():
            match = version_dir_pattern.match(entry.name)
            if match:
                step = int(match.group(1))
                step2path[step] = os.path.join(entry.path, "results.json")

    for step, results_path in step2path.items():
        data['step'].append(step)
        if os.path.isfile(results_path):
            with open(results_path, "r") as f:
                results_json = json.load(f)
            metrics_dict = results_json.get("metrics", {})
        else:
            metrics_dict = {}

        # Fill in metric values
        for m in metrics:
            value = metrics_dict.get(m, None)
            data[m].append(value)
    
    # check which solutions are valid and print them out
    for key, values in data.items():
        if key == 'is_valid':
            true_indices = [i for i, value in enumerate(values) if value]
            if true_indices:
                print(f"Indices of valid versions: {true_indices}")
                
    df = pd.DataFrame(data)
    df.sort_values(by="step", inplace=True, ignore_index=True)

    return df


def plot_gap_comparison(data_dicts, figsize=(10, 5), main_title='Recovered Time Percentage Comparison With Paper-Like Prompt'):
    """
    Create a subplot visualization comparing gap percentages with adaptive layout.
    
    Args:
        data_dicts: List of tuples containing (dictionary, title, color)
        figsize: Tuple of figure size (width, height)
        main_title: Main title for the entire figure
    """
    # Calculate the number of rows and columns for the subplot grid
    n_plots = len(data_dicts)
    n_cols = math.ceil(math.sqrt(n_plots))  # Square root for balanced layout
    n_rows = math.ceil(n_plots / n_cols)
    
    # Adjust figure size based on number of subplots
    width, height = figsize
    figsize = (width * (n_cols/2), height * (n_rows/2))
    
    # Create subplot layout
    fig, axs = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_plots == 1:
        axs = np.array([axs])
    axs = axs.flatten()  # Flatten the array to make indexing easier

    # Create a barplot for each dictionary
    for i, (data_dict, title, color) in enumerate(data_dicts):
        # Sort the dictionary by keys
        sorted_items = sorted(data_dict.items(), key=lambda x: int(x[0]))
        keys = [item[0] for item in sorted_items]
        values = [item[1] for item in sorted_items]
        
        # Create the barplot with the specified color
        bars = axs[i].bar(range(len(keys)), values, color=color, edgecolor='black', alpha=0.8)
        
        # Add title and labels
        axs[i].set_title(title, fontsize=14, fontweight='bold')
        axs[i].set_xlabel('Record Number', fontsize=12)
        axs[i].set_ylabel('Recovered Time (s)', fontsize=12)
        
        # Ensure all xticks are shown
        axs[i].set_xticks(range(len(keys)))
        axs[i].set_xticklabels(keys, fontsize=10)
        
        # Add grid for better readability
        axs[i].grid(True, linestyle='--', alpha=0.7)
        
        # Add a horizontal line at y=0
        axs[i].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Format y-axis as percentage
        axs[i].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))
        
        # Annotate the values on top of the bars
        for bar in bars:
            height = bar.get_height()
            if height >= 0:
                y_pos = height + 0.02
            else:
                y_pos = height - 0.05
            axs[i].text(
                bar.get_x() + bar.get_width()/2.,
                y_pos,
                '{:.1%}'.format(height),
                ha='center', 
                fontsize=9,
                fontweight='bold',
                color='black'
            )
    
    # Hide empty subplots if any
    for i in range(n_plots, len(axs)):
        axs[i].set_visible(False)

    # Add a main title for the figure
    fig.suptitle(main_title, fontsize=16, fontweight='bold', y=0.98)

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make room for the suptitle

    return fig, axs