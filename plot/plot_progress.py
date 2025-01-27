import argparse
import os
import re
import json
import pandas as pd
import matplotlib.pyplot as plt


def gather_metrics(workspace_path: str, metrics: list[str]) -> pd.DataFrame:
    data = {'step': []}
    for m in metrics:
        data[m] = []

    # Gather metrics from version subdirectories
    version_dir_pattern = re.compile(r"^v_(\d+)$")
    for entry in os.scandir(workspace_path):
        if entry.is_dir():
            match = version_dir_pattern.match(entry.name)
            if match:
                step = int(match.group(1))
                data['step'].append(step)

                results_path = os.path.join(entry.path, "results.json")
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


def main():
    parser = argparse.ArgumentParser(description="Plot progress of a single metric across versioned directories.")
    parser.add_argument('--workspace_path', type=str, required=True,
                        help="Path to the workspace, e.g. workspaces/task_<timestamp>")
    parser.add_argument('--metric', type=str, required=True,
                        help="Name of the metric to plot, e.g. 'accuracy'")
    args = parser.parse_args()

    # Gather data into DataFrame
    df = gather_metrics(args.workspace_path, [args.metric])

    if df.empty:
        print("No data found for the specified metric.")
        return

    # Plot the metric over step
    fig, ax = plt.subplots()
    ax.plot(df['step'], df[args.metric], marker='o', linestyle='-')
    ax.set_xlabel("Iteration")
    ax.set_ylabel(args.metric)

    ax.grid(color='black', linestyle='-', linewidth=1, alpha=0.2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
