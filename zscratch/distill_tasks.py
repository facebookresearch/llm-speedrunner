"""Find subset of up to K task that are most predictive of the remaining tasks' performance.

Usage:

python zscratch/distill_tasks.py \
--mlebench_cache_path='~/research/reference/mle-bench/runs/data.pkl' \
--cache_path='results/distill_mlebench_results_subset3.pkl' \
--max_subset_size=3 \
--sort_by=pearson_r \
--log_norm \
--n_plots=5 \
--render_plots \
--save_plot_filename=distill_tasks_subset3.png
"""

import argparse
import itertools
import math
import os
import warnings

import pandas as pd
import numpy as np
from scipy.stats import spearmanr, ConstantInputWarning
from tqdm import tqdm
import matplotlib.pyplot as plt


warnings.filterwarnings("ignore", category=ConstantInputWarning)


pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_columns', None)  # Show all columns


def plot_top_bottom(results_df_sorted, n_plots, sort_by, save_filename=None):
    # Ensure we don't try to plot more runs than available.
    n_plots = min(n_plots, len(results_df_sorted) // 2)  
    top = results_df_sorted.head(n_plots)
    nonan_results = results_df_sorted[results_df_sorted[sort_by].notna()]
    n_select_bottom = min(n_plots, len(nonan_results))
    bottom = nonan_results.tail(n_select_bottom)
    
    fig, axes = plt.subplots(nrows=2, ncols=n_plots, figsize=(5 * n_plots, 10))
    
    # If only one plot per row, axes may not be 2D.
    if n_plots == 1:
        axes = np.array([[axes[0]], [axes[1]]])
    
    # Plot top runs in the first row.
    for i, (_, row) in enumerate(top.iterrows()):
        ax = axes[0, i]
        X = np.array(row['X_data'])
        Y = np.array(row['Y_data'])
        slope = row['slope']
        intercept = row['intercept']
        pearson_r = row['pearson_r']
        sort_value = row[sort_by]
        
        ax.scatter(X, Y, color='blue')
        if not np.isnan(slope) and len(X) > 0:
            x_min, x_max = np.nanmin(X), np.nanmax(X)
            x_line = np.array([x_min, x_max])
            y_line = slope * x_line + intercept
            label = f'slope={slope:.2f}, r={pearson_r:.3f}'
            if sort_by != 'pearson_r':
                label += f' {sort_by}={sort_value:.3f}'
            ax.plot(x_line, y_line, color='red', label=label)
        
        title_text = "\n".join(row['competition_ids'].split())
        ax.set_title(title_text)
        ax.set_xlabel('Mean normalized score on subset')
        ax.set_ylabel('Mean normalized score on complement')
        ax.legend()
    
    # Plot bottom runs in the second row.
    for i, (_, row) in enumerate(bottom.iterrows()):
        ax = axes[1, i]
        X = np.array(row['X_data'])
        Y = np.array(row['Y_data'])
        slope = row['slope']
        intercept = row['intercept']
        pearson_r = row['pearson_r']
        sort_value = row[sort_by]
        
        ax.scatter(X, Y, color='blue')
        if not np.isnan(slope) and len(X) > 0:
            x_min, x_max = np.nanmin(X), np.nanmax(X)
            x_line = np.array([x_min, x_max])
            y_line = slope * x_line + intercept
            label = f'slope={slope:.2f}, r={pearson_r:.3f}'
            if sort_by != 'pearson_r':
                label += f' {sort_by}={sort_value:.3f}'
            ax.plot(x_line, y_line, color='red', label=label)
        
        title_text = "\n".join(row['competition_ids'].split())
        ax.set_title(title_text)
        ax.set_xlabel('Mean normalized score on subset')
        ax.set_ylabel('Mean normalized score on complement')
        ax.legend()
    
    plt.tight_layout()

    if save_filename:
        plt.savefig(f'figures/{save_filename}')

    plt.show()


def main():
    # Parse command-line arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("--mlebench_cache_path", type=str,
                        help="Path to the pickle file containing experiment and competition dataframes")
    parser.add_argument("--cache_path", type=str,
                        help="Path to the pickle file of previously computed regression results.")
    parser.add_argument("--max_subset_size", type=int, default=2,
                        help="Max task subset to consider.")
    parser.add_argument("--log_norm", action='store_true', help="Log normalize the scores.")
    parser.add_argument("--sort_by", type=str, default='pearson_r', choices=['pearson_r', 'spearman_r'],
                        help="Sort results (and plot top k) based on this measure.")
    parser.add_argument("--n_plots", type=int, default=4,
                        help="Number of top and bottom runs to track and plot")
    parser.add_argument("--render_plots", action='store_true',
                        help="Flag to render plots for the top and bottom runs in a single figure")
    parser.add_argument("--save_plot_filename", type=str, default=None, help="Save plot here.")
    args = parser.parse_args()

    # Load the pickle file containing two dataframes: reports_df and competition_df.
    data = pd.read_pickle(args.mlebench_cache_path)
    results_df = None
    if args.cache_path and os.path.exists(args.cache_path):
        results_df = pd.read_pickle(args.cache_path)

    # Expecting data to be a dict with keys 'reports_df' and 'competition_df'
    reports_df = data['reports_df']
    competition_df = data['competition_df']

    if results_df is None:
        merged_df = reports_df.merge(competition_df, on='competition_id', how='left')
        merged_df['lower_is_better'] = merged_df['gold_threshold'] < merged_df['bronze_threshold']

        merged_df['normalized_score'] = (
            (merged_df['score'] - merged_df['median_threshold']) /
            (merged_df['gold_threshold'] - merged_df['median_threshold'])
        )
        merged_df.fillna(0, inplace=True)

        if args.log_norm:
            merged_df['normalized_score'] = np.log(1 + np.maximum(0, merged_df['normalized_score']))
        else:
            merged_df['normalized_score'] = np.maximum(0, merged_df['normalized_score'])

        # Group by (experiment_id, competition_id) and compute aggregate statistics on the normalized score.
        grouped = merged_df.groupby(['experiment_id', 'competition_id'])
        agg_stats = grouped['normalized_score'].agg(['mean', 'median', 'min', 'max', 'std', 'count']).reset_index()
        agg_stats.rename(columns={
            'mean': 'mean_score',
            'median': 'median_score',
            'min': 'min_score',
            'max': 'max_score',
            'std': 'std_score',
            'count': 'total_count'
        }, inplace=True)

        # Calculate the standard error of the mean (sem_score)
        agg_stats['sem_score'] = agg_stats['std_score'] / np.sqrt(agg_stats['total_count'])
        agg_df = agg_stats

        # Prepare to store regression results over subsets of competition_ids.
        comp_ids = np.sort(agg_df['competition_id'].unique())
        results = []
        max_subset_size = min(args.max_subset_size, len(comp_ids))
        
        # Iterate over all non-empty subsets of competition_ids (of sizes 1 to max_subset_size)
        for r in range(1, max_subset_size + 1):
            total_combos = math.comb(len(comp_ids), r)
            for subset in tqdm(itertools.combinations(comp_ids, r),
                               total=total_combos,
                               desc=f'Computing regressions for subsets of size {r}'):
                subset_set = set(subset)
                complement_set = set(comp_ids) - subset_set

                # Compute weighted mean performance for each experiment_id in the chosen subset.
                subset_df = agg_df[agg_df['competition_id'].isin(subset_set)]
                if subset_df.empty:
                    continue
                weighted_subset = subset_df.groupby('experiment_id')[['mean_score', 'total_count']].apply(
                    lambda g: np.sum(g['mean_score'] * g['total_count']) / g['total_count'].sum() 
                              if g['total_count'].sum() != 0 else 1
                ).reset_index(name='subset_mean')

                # Compute weighted mean performance for the complement competition_ids.
                complement_df = agg_df[agg_df['competition_id'].isin(complement_set)]
                if complement_df.empty:
                    continue
                weighted_complement = complement_df.groupby('experiment_id')[['mean_score', 'total_count']].apply(
                    lambda g: np.sum(g['mean_score'] * g['total_count']) / g['total_count'].sum() 
                              if g['total_count'].sum() != 0 else 1
                ).reset_index(name='complement_mean')

                # Merge the subset and complement results on experiment_id.
                merged_weighted = pd.merge(weighted_subset, weighted_complement, on='experiment_id', how='inner')
                if merged_weighted.empty:
                    continue

                # Extract the paired performance measures.
                X = merged_weighted['subset_mean'].values
                Y = merged_weighted['complement_mean'].values

                # Count the number of valid X, Y pairs (neither value is nan)
                valid_mask = (~np.isnan(X)) & (~np.isnan(Y))
                total_valid_points = int(np.sum(valid_mask))

                # Perform linear regression with no intercept.
                if np.nansum(X**2) == 0 or np.nansum(Y**2) == 0:
                    slope = np.nan
                    r2 = np.nan
                else:
                    slope = np.nansum(X * Y) / np.nansum(X**2)
                    predictions = slope * X
                    r2 = 1 - np.nansum((Y - predictions) ** 2) / np.nansum(Y**2)

                # Compute Pearson correlation coefficient
                X_mean = np.nanmean(X)
                Y_mean = np.nanmean(Y)
                denom_X = np.nansum((X - X_mean)**2)
                denom_Y = np.nansum((Y - Y_mean)**2)
                cov_XY = np.nansum((X - X_mean) * (Y - Y_mean))

                if denom_X == 0 or denom_Y == 0:
                    r = np.nan
                    slope = np.nan
                    intercept = np.nan
                else:
                    r = cov_XY / np.sqrt(denom_X * denom_Y)
                    slope = cov_XY / denom_X
                    intercept = Y_mean - slope * X_mean

                spearman_corr, _ = spearmanr(X, Y)

                # Store the subset (as a sorted list in string form) and the computed metrics,
                # along with the actual X, Y datapoints, slope, and count of valid points.
                results.append({
                    'competition_ids': ', '.join(sorted(list(subset_set))),
                    'pearson_r': r,
                    'spearman_r': spearman_corr,
                    'X_data': X.tolist(),
                    'Y_data': Y.tolist(),
                    'total_valid_points': total_valid_points,
                    'slope': slope,
                    'intercept': intercept
                })

        results_df = pd.DataFrame(results)

        if args.cache_path and not os.path.exists(args.cache_path):
            cache_dir_path = os.path.dirname(args.cache_path)
            os.makedirs(cache_dir_path, exist_ok=True)
            results_df.to_pickle(args.cache_path)

    # Sort by pearson_r to extract top and bottom runs.
    results_df_sorted = results_df.sort_values(by=args.sort_by, ascending=False)
    n_select = min(args.n_plots, len(results_df_sorted))
    top_k = results_df_sorted.head(n_select)
    nonan_results = results_df_sorted[results_df_sorted[args.sort_by].notna()]
    n_select_bottom = min(args.n_plots, len(nonan_results))
    bottom_k = nonan_results.tail(n_select_bottom)

    print(f"Top 10 runs by {args.sort_by}:")
    print(top_k[['competition_ids', args.sort_by, 'total_valid_points']])
    print(f"\nBottom 10 runs by {args.sort_by}:")
    print(bottom_k[['competition_ids', args.sort_by, 'total_valid_points']])

    # Render plots if the flag is set.
    if args.render_plots:
        plot_top_bottom(results_df_sorted, args.n_plots, args.sort_by, args.save_plot_filename)


if __name__ == '__main__':
    main()
