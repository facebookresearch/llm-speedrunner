"""Plot MLE-Bench results

Usage:

(Run from project root)

python -m plot.summarize_mlebench \
--runs_dir_path='~/research/reference/mle-bench/runs' \
--cache_path='~/research/reference/mle-bench/runs/data.pkl' \
--save_filename=mlebench_lite_results.pdf


python -m plot.summarize_mlebench \
--runs_dir_path='~/research/reference/mle-bench/runs' \
--cache_path='~/research/reference/mle-bench/runs/data.pkl' \
--plot_type=pass_at_k \
--save_filename=mlebench_lite_pass_at_k.pdf
"""

import os
import glob
import json
import math
import pickle
import re
import seaborn as sns
import tqdm

from typing import Optional
from tqdm import tqdm
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from utils import fs_utils

# ------------------------------------------------------------------
# Load from a Single Pickle File (if it exists) or regenerate
# ------------------------------------------------------------------

DATA_PKL = 'data.pkl'

reports_df = competition_df = None


def load_dfs(runs_dir_path: str, cache_path: Optional[str] = None):
    global reports_df, competition_df

    if os.path.exists(cache_path):
        print('Loading existing data from pickle file...')
        with open(cache_path, 'rb') as f:
            data_dict = pickle.load(f)
        reports_df = data_dict.get('reports_df')
        competition_df = data_dict.get('competition_df')
    else:
        print('Pickle file not found. Processing CSV and JSON files...')
        csv_path = os.path.join(runs_dir_path, 'run_group_experiments.csv')
        try:
            df_experiments = pd.read_csv(csv_path)
        except Exception as e:
            raise RuntimeError(f'Error reading CSV file "{csv_path}": {e}')
        
        experiment_run_groups = {}
        for _, row in df_experiments.iterrows():
            exp_id = str(row['experiment_id'])
            run_group = str(row['run_group'])
            experiment_run_groups.setdefault(exp_id, []).append(run_group)
        
        report_rows = []
        competition_dict = {}  # one unique row per competition_id

        for experiment_id, run_groups in tqdm(experiment_run_groups.items(), desc='Processing experiments', unit='exp'):
            for run_group in tqdm(run_groups, desc='Processing run groups', leave=False, unit='folder'):
                run_group_dir_path = os.path.join(runs_dir_path, run_group)

                if not os.path.isdir(run_group_dir_path):
                    print(f'[Warning] Folder "{run_group_dir_path}" does not exist. Skipping.')
                    continue
                
                json_files = glob.glob(os.path.join(run_group_dir_path, '*.json'))
                for json_file in tqdm(json_files, desc='Processing JSON files', leave=False, unit='file'):
                    try:
                        with open(json_file, 'r') as f:
                            data = json.load(f)
                    except Exception as e:
                        print(f'[Error] Failed to load "{json_file}": {e}')
                        continue
                    
                    if 'competition_reports' not in data:
                        continue
                    
                    for report in data['competition_reports']:
                        report_row = {
                            'experiment_id': experiment_id,
                            'competition_id': report.get('competition_id'),
                            'score': report.get('score'),
                            'any_medal': report.get('any_medal'),
                            'gold_medal': report.get('gold_medal'),
                            'silver_medal': report.get('silver_medal'),
                            'bronze_medal': report.get('bronze_medal'),
                            'above_median': report.get('above_median'),
                            'valid_submission': report.get('valid_submission')
                        }
                        report_rows.append(report_row)
                        
                        comp_id = report.get('competition_id')
                        if comp_id not in competition_dict:
                            competition_dict[comp_id] = {
                                'competition_id': comp_id,
                                'gold_threshold': report.get('gold_threshold'),
                                'silver_threshold': report.get('silver_threshold'),
                                'bronze_threshold': report.get('bronze_threshold'),
                                'median_threshold': report.get('median_threshold')
                            }
        
        reports_df = pd.DataFrame(report_rows)
        competition_df = pd.DataFrame(list(competition_dict.values()))
        
        data_dict = {'reports_df': reports_df, 'competition_df': competition_df}

        if cache_path is not None:
            with open(cache_path, 'wb') as f:
                pickle.dump(data_dict, f)
            
        print('Processing complete and data saved to pickle file.')

    print('Reports DataFrame:')
    print(reports_df.head())
    print('\nCompetition DataFrame:')
    print(competition_df.head())

# ------------------------------------------------------------------
# Utility Functions
# ------------------------------------------------------------------

def get_submission_stats_for_competition_id(experiment_id: str, competition_id: str) -> dict:
    """
    For a given experiment_id and competition_id, compute the mean values of key metrics.
    Also count the total submissions for these filters.
    Returns a dict with this info.
    """
    global reports_df, competition_df

    mask = (reports_df['experiment_id'] == experiment_id) & (reports_df['competition_id'] == competition_id)
    filtered = reports_df[mask]
    total_submissions = len(filtered)
    
    if total_submissions == 0:
        print(f'[Info] No submissions for experiment_id "{experiment_id}"" and competition_id "{competition_id}".')
        return {
            'experiment_id': experiment_id,
            'competition_id': competition_id,
            'total_submissions': 0,
            'score': None,
            'any_medal': None,
            'gold_medal': None,
            'silver_medal': None,
            'bronze_medal': None,
            'above_median': None,
            'valid_submission': None,
        }
        
    keys = ['score', 'any_medal', 'gold_medal', 'silver_medal', 'bronze_medal', 'above_median', 'valid_submission']
    mean_values = filtered[keys].mean(skipna=True).to_dict()
    err_values = {f'{k}_err': v for k, v in filtered[keys].sem(skipna=True).to_dict().items()}
    stats = {
        'experiment_id': experiment_id,
        'competition_id': competition_id,
        'total_submissions': total_submissions
    }
    stats.update(mean_values)
    stats.update(err_values)
    return stats

def get_experiment_summary(experiment_ids: str | list[str], competition_ids: list[str]=None, group_label=None) -> list:
    """
    Computes summary statistics for one or more experiment_ids.
    If competition_ids is provided, only include those competitions; otherwise include all.
    Returns a list of dictionaries—one for each distinct experiment_id/competition_id pair.
    (A global summary row with competition_id 'all' is included for each experiment.)
    """
    global reports_df, competition_df

    # Allow experiment_ids to be a single string or a list
    if not isinstance(experiment_ids, list):
        experiment_ids = [experiment_ids]
    
    summaries = []
    for exp_id in experiment_ids:
        exp_data = reports_df[reports_df['experiment_id'] == exp_id]
        if competition_ids is not None:
            exp_data = exp_data[exp_data['competition_id'].isin(competition_ids)]
        
        if len(competition_ids) > 1:
            overall_total = len(exp_data)
            if overall_total == 0:
                print(f'[Info] No submissions found for experiment_id "{exp_id}".')
                continue
            
            keys = ['score', 'any_medal', 'gold_medal', 'silver_medal', 'bronze_medal', 'above_median', 'valid_submission']
            overall_mean = exp_data[keys].mean().to_dict()
            overall_err = {f'{k}_err': v for k, v in exp_data[keys].sem().to_dict().items()}
            overall_stats = {
                'experiment_id': exp_id,
                'competition_id': 'all' if group_label is None else group_label,
                'total_submissions': overall_total
            }
            overall_stats.update(overall_mean)
            overall_stats.update(overall_err)
            summaries.append(overall_stats)
        
        # Compute per-competition stats
        comp_ids = competition_ids if competition_ids is not None else exp_data['competition_id'].unique()
        for comp_id in comp_ids:
            stats = get_submission_stats_for_competition_id(exp_id, comp_id)
            summaries.append(stats)
    
    return summaries

def plot_summary(summary_list: list[dict], keys: list[str] = None, ncols: int = 4, save_filename: Optional[str] = None):
    """Plot a list of summary info dicts returned by the above.
    Args:
      - summary_list: list of summary dicts.
      - keys: list of keys to plot on the x-axis. If None, defaults to numeric keys (excluding 'experiment_id' and 'competition_id').
      - ncols: Number of subplot columns.
    """
    global reports_df, competition_df

    if not summary_list:
        print('No summaries provided to plot.')
        return

    if keys is None:
        sample = summary_list[0]
        keys = [k for k, v in sample.items() if k not in ['experiment_id', 'competition_id'] and isinstance(v, (int, float))]
    
    grouped = defaultdict(list)
    for s in summary_list:
        comp = s.get('competition_id', 'all')
        grouped[comp].append(s)
    
    comp_ids = list(grouped.keys())
    n_plots = len(comp_ids)
    nrows = (n_plots + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 4 * nrows))
    
    if n_plots == 1:
        axes = [axes]
    elif nrows == 1:
        axes = axes.flatten()
    else:
        axes = [ax for row in axes for ax in row]
    
    # Determine unique experiment_ids (for consistent coloring)
    unique_experiments = sorted({s['experiment_id'] for s in summary_list})
    cmap = sns.color_palette('husl', len(unique_experiments), as_cmap=True)
    colors = {exp: cmap(i / len(unique_experiments)) for i, exp in enumerate(unique_experiments)}
    
    # Mapping for threshold keys (if applicable)
    threshold_keys = {
        'gold_medal': ('gold_threshold', 'gold'),
        'silver_medal': ('silver_threshold', 'silver'),
        'bronze_medal': ('bronze_threshold', 'brown'),
        'above_median': ('median_threshold', 'darkgreen')
    }
    
    for idx, comp_id in enumerate(comp_ids):
        ax = axes[idx]
        group_data = grouped[comp_id]
        n_experiments = len(group_data)
        
        x = np.arange(len(keys))
        bar_width = 0.8 / n_experiments
        
        # Plot bars for each experiment_id
        for i, summary in enumerate(group_data):
            exp_id = summary['experiment_id']
            values = [summary.get(k, 0) for k in keys]
            errs = np.array([summary.get(f'{k}_err', 0) for k in keys]) 
            offset = (i - (n_experiments - 1) / 2) * bar_width
            ax.bar(x + offset, values, width=bar_width, color=colors[exp_id],
                   label=exp_id if comp_id == 'all' else None,
                   yerr=errs, capsize=5, error_kw={'elinewidth': 1, 'capsize': 5})
        
        ax.set_title(f'{comp_id}')
        ax.set_xticks(x)
        ax.set_xticklabels(keys, rotation=45, ha='right')
        ax.set_ylabel('Mean Value')
        
        # Add thresholds if competition-specific score is present
        if comp_id != 'all' and 'score' in keys:
            comp_info = competition_df[competition_df['competition_id'] == comp_id]
            if not comp_info.empty:
                thresholds = comp_info.iloc[0].to_dict()
                # For each key that has a threshold, draw a horizontal dashed line
                for metric, (thresh_key, color) in threshold_keys.items():
                    if metric in keys:
                        thresh_value = thresholds.get(thresh_key)
                        if thresh_value is not None:
                            ax.axhline(y=thresh_value, color=color, linestyle='--',
                                       label=f'{thresh_key}: {thresh_value}')
        
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend()
    
    # Remove unused subplots.
    for j in range(n_plots, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    
    if save_filename:
        plt.savefig(f'figures/{save_filename}', dpi=300, bbox_inches='tight') 

    plt.show()


def pass_at_k(n, c, k):
    """
    Args:
        n: total number of samples
        c: number of correct samples
        k: k in pass@$k$
    """
    if n - c < k: return 1.0
    return 1.0 - np.prod(1.0 - k /np.arange(n - c + 1, n + 1))


def get_pass_at_k_stats(experiment_ids: str | list[str],
                        competition_ids: list[str] = None,
                        keys: list[str] = None,
                        k: Optional[int] = None):
    """
    Args:
        experiment_ids: either a single experiment_id (str) or list of them.
        competition_ids: if provided, only include rows with these competition_id values.
        keys: list of column names (metrics) to compute pass@k for. If None, use defaults.
        k: the k to use in pass@k (default is 1).

    Returns:
        A new dataframe with columns: experiment_id, competition_id, and for each key, the pass@k value.
    """
    global reports_df

    if isinstance(experiment_ids, str):
        experiment_ids = [experiment_ids]

    df = reports_df[reports_df['experiment_id'].isin(experiment_ids)]

    if competition_ids is None:
        competition_list = sorted(df['competition_id'].unique())
    else:
        competition_list = competition_ids

    if keys is None:
        keys = [
            'gold_medal', 
            'silver_medal',
            'bronze_medal',
            'any_medal',
            'above_median'
        ]

    grouped = df.groupby(['experiment_id', 'competition_id'])

    # Compute the maximum number of runs among all groups (0 for runs under this number)
    max_runs = grouped.size().max()
    if k is None:
        k = max_runs

    agg_df = grouped.agg({col: 'sum' for col in keys})

    # Fill missing values with 0
    full_index = pd.MultiIndex.from_product([experiment_ids, competition_list],
                                              names=['experiment_id', 'competition_id'])
    agg_df = agg_df.reindex(full_index, fill_value=0)

    # Now compute pass@k for 1...k using the maximum number of runs as n
    for col in tqdm(keys):
        for cur_k in range(1, k + 1):
            new_col = f'{col}_pass_at_{cur_k}'
            agg_df[new_col] = agg_df[col].apply(lambda c: pass_at_k(max_runs, c, cur_k))

    agg_df = agg_df.drop(columns=keys)

    agg_df = agg_df.reset_index()

    return agg_df


def plot_pass_at_k(df, ncols=4, save_filename: Optional[str] = None):
    """
    Plots pass@k curves based on the aggregated dataframe produced earlier.
    
    Parameters:
        df (pd.DataFrame): The aggregated dataframe (agg_df) that has columns 
                           'experiment_id', 'competition_id', and for each metric a 
                           series of columns named '{key}_pass_at_{k}'.
        ncols (int): Number of columns in the subplot grid. (Default is 4.)
        
    Plot layout:
      - The very first subplot is reserved for a custom legend only.
      - The remaining subplots each correspond to one competition_id.
      - The x-axis in each competition plot is k (from 1 to max k).
      - The y-axis is the pass@k metric.
      - Each experiment_id (method) gets its own color (using a Husl palette).
      - Each key (metric) gets its own marker shape.
      - Only the first subplot shows a legend; the rest do not.
    """
    # Get unique experiments and competitions
    unique_experiments = df['experiment_id'].unique()
    unique_competitions = sorted(df['competition_id'].unique())
    
    # Extract keys and k values from the columns. 
    # We assume columns are like: '<key>_pass_at_<k>'.
    pattern = re.compile(r'^(.*?)_pass_at_(\d+)$')
    keys = set()
    ks = set()
    for col in df.columns:
        m = pattern.match(col)
        if m:
            keys.add(m.group(1))
            ks.add(int(m.group(2)))
    keys = sorted(keys)
    max_k = max(ks) if ks else 0

    # Unique color per experiment_id
    colors = sns.color_palette('husl', len(unique_experiments))
    color_dict = {exp: colors[i] for i, exp in enumerate(unique_experiments)}
    
    # Unique markers per key
    markers_list = ['o', 's', 'D', 'X', '^', 'v', '<', '>', 'P', '*']
    marker_dict = {key: markers_list[i % len(markers_list)] for i, key in enumerate(keys)}
    
    # Set up subplots
    n_subplots = len(unique_competitions) + 1
    nrows = math.ceil(n_subplots / ncols)
    
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*6, nrows*4))
    if nrows * ncols > 1:
        axs = axs.flatten()
    else:
        axs = [axs]
    
    # Custom legend in first subplot
    ax_leg = axs[0]
    ax_leg.axis('off')
    legend_elements = []
    for exp in unique_experiments:
        legend_elements.append(Line2D([0], [0], marker='o', color='w',
                                      markerfacecolor=color_dict[exp],
                                      markersize=10, label=f'{exp}'))
    if len(keys) > 1:
        for key in keys:
            legend_elements.append(Line2D([0], [0], marker=marker_dict[key], color='w',
                                          markerfacecolor='darkgray',
                                          markersize=10, label=f'Key: {key}'))
    else:
        fig.suptitle(f'pass@$k$, {keys[0]}', fontsize=20, y=1.0)

    # Add the legend to the first subplot.
    ax_leg.legend(handles=legend_elements, loc='center', frameon=False, fontsize=18)
    
    # Per competition subplots
    for i, comp in enumerate(unique_competitions):
        ax = axs[i+1]  # skip index 0 (reserved for legend)
        # Filter rows for this competition_id.
        comp_df = df[df['competition_id'] == comp]
        # For each row (each experiment_id in this competition), plot each key's pass@k.
        for _, row in comp_df.iterrows():
            exp = row['experiment_id']
            for key in keys:
                # Build lists of k values (x) and corresponding pass@k values (y)
                x_vals = list(range(1, max_k+1))
                y_vals = []
                for k_val in x_vals:
                    col_name = f'{key}_pass_at_{k_val}'
                    y_vals.append(row.get(col_name, np.nan))
                # Plot the curve: color is based on experiment, marker is based on key.
                ax.plot(x_vals, y_vals, label=f'{exp}-{key}', 
                        color=color_dict[exp],
                        marker=marker_dict[key], linestyle='-', markersize=6)
        ax.set_title(f'{comp}', fontsize=14)
        ax.set_xlabel('k', fontsize=14)
        ax.set_ylabel('pass@k', fontsize=14)
        ax.tick_params(axis='both', labelsize=12)
        ax.set_xticks(range(1, max_k+1))
        # Turn off the legend for these subplots.
        if ax.get_legend() is not None:
            ax.get_legend().remove()
    
    # Hide any extra axes that were created but not used.
    for ax in axs[len(unique_competitions)+1:]:
        ax.axis('off')
    
    plt.tight_layout()

    if save_filename:
        plt.savefig(f'figures/{save_filename}', dpi=300, bbox_inches='tight') 

    plt.show()


if __name__ == '__main__':
    import argparse

    import mlebench
    from mlebench.registry import Registry

    parser = argparse.ArgumentParser(description='Plot progress of a single metric across versioned directories.')
    parser.add_argument('--runs_dir_path', type=str, required=True, help='Path to MLE-Bench runs directory')
    parser.add_argument('--cache_path', type=str, default=None, help='Path to cached data frames')
    parser.add_argument('--save_filename', type=str, default=None, help='Path to save plot')
    parser.add_argument('--plot_type', type=str, default='success_rate', choices=['success_rate', 'pass_at_k'], help='Type of plot')
    parser.add_argument('--pass_at_k', type=int, default=None, help='Pass at k for this value of k')
    args = parser.parse_args()
    
    # Can be string or list of strings
    experiment_ids = [
        'models-o1-preview-aide',
        'biggpu-gpt4o-aide',
        'cpu-gpt4o-aide',
        'models-claude35sonnet-aide',
        'models-llama-3.1-405B-instruct-aide',
    ]
        
    runs_dir_path = fs_utils.expand_path(args.runs_dir_path)
    cache_path = None
    if args.cache_path is not None:
        cache_path = fs_utils.expand_path(args.cache_path)
    load_dfs(runs_dir_path, cache_path)

    # competition_ids = None  # Use all competitions
    competition_ids = Registry().get_lite_competition_ids()

    # plot these metrics
    # keys = [
    #     'above_median', 'bronze_medal', 'silver_medal', 'gold_medal', 'any_medal'
    # ]

    keys = [
        'gold_medal'
    ]

        
    if args.plot_type == 'success_rate':
        # Get summary info (one row per distinct experiment_id, competition_id pair)
        summary = get_experiment_summary(
            experiment_ids,
            competition_ids=competition_ids,
            group_label='lite'
        )

        if summary:
            plot_summary(summary, keys=keys, ncols=4, save_filename=args.save_filename)
    elif args.plot_type == 'pass_at_k':
        pass_at_k_df = get_pass_at_k_stats(experiment_ids, competition_ids, keys=keys, k=args.pass_at_k)
        plot_pass_at_k(pass_at_k_df, save_filename=args.save_filename)
    else:
        raise ValueError(f'Unrecognized plot_type {args.plot_type}')
