"""Plot progress in a metric vs. experiment iterations

Usage:
python plot/plot_progress.py \
--workspace_template_path=workspace_templates/collatz \
--workspace_path=workspaces/collatz_test2 \
--metric=max_steps \
--ylabel='Max Collatz sequence length' \
--save_name=collatz_max_steps.pdf

"""
from typing import Optional

import argparse
import os
import re
import json

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
from matplotlib.ticker import MaxNLocator


rcParams['font.size'] = 14
rcParams['figure.figsize'] = [12, 6]


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
    
    df = pd.DataFrame(data)
    df.sort_values(by="step", inplace=True, ignore_index=True)

    return df


def annotate_with_labels(points, labels, xyoffset=(4, 16)):
    for (x, y), label in zip(points, labels):
        plt.annotate(
            label, 
            (x, y),
            xytext=xyoffset,
            textcoords='offset pixels',
            ha='right',
            va='bottom',
            fontsize=12,
        )

    plt.scatter(*tuple(zip(*points)), color='red', zorder=5)


def get_step_labels_for_collatz():
    return [
        (1, 'Add memoization'),
        (2, 'Batch even sequences'),
        (3, 'Binary jumping + multi-threading'),
    ]


def get_point_labels(steps: int, labels: str, ys: pd.DataFrame):
    points = [(step, ys.iloc[step]) for step in steps]
    return {
        'points': points,
        'labels': labels
    }



def main():
    parser = argparse.ArgumentParser(description="Plot progress of a single metric across versioned directories.")
    parser.add_argument('--workspace_template_path', type=str, required=True,
                        help="Path to the workspace, e.g. workspace_templates/collatz")
    parser.add_argument('--workspace_path', type=str, required=True,
                        help="Path to the workspace, e.g. workspaces/task_<timestamp>")
    parser.add_argument('--metric', type=str, required=True,
                        help="Name of the metric to plot, e.g. 'accuracy'")
    parser.add_argument('--xlabel', type=str, default='Iteration',
                        help="ylabel, defaults to 'Iteration'")
    parser.add_argument('--ylabel', type=str, default=None,
                        help="ylabel, defaults to metric")
    parser.add_argument('--yrescale', type=float, default=None,
                        help="Value by which to rescale the y-values")
    parser.add_argument('--ythreshold', type=float, default=None,
                        help="Plot y-threshold value here")
    parser.add_argument('--ignore_threshold', type=float, default=None,
                        help="Remove datapoints that equal this on the y-axis.")
    parser.add_argument('--save_name', type=str, default=None,
                        help="Save to figures/<save_name>. Should include the file extension, e.g. .pdf")
    args = parser.parse_args()

    # Gather data into DataFrame
    df = gather_metrics(args.workspace_path, [args.metric], args.workspace_template_path)

    if args.ignore_threshold is not None:
        df.loc[df[args.metric] > args.ignore_threshold, args.metric] = np.nan

    if df.empty:
        print("No data found for the specified metric.")
        return

    if args.yrescale:
        df[args.metric] = df[args.metric]*args.yrescale

    # Plot the metric over experiment iterations
    fig, ax = plt.subplots()
    ax.plot(df['step'], df[args.metric], marker='o', linestyle='-')
    ax.set_xlabel(args.xlabel)
    ax.set_ylabel(args.ylabel or args.metric)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

    if args.ythreshold is not None:
        plt.axhline(y=args.ythreshold, color='orange', linestyle='-', linewidth=2)


    # Set some hypotheses here
    # if 'collatz' in args.workspace_template_path:
    #     annotate_with_labels(
    #         **get_point_labels(
    #             *tuple(zip(*get_step_labels_for_collatz())), df[args.metric], 
    #         )
    #     )

    ax.grid(color='black', linestyle='-', linewidth=1, alpha=0.1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()

    if args.save_name:
        plt.savefig(f'figures/{args.save_name}', dpi=300, bbox_inches='tight')

    plt.show()


if __name__ == "__main__":
    main()
