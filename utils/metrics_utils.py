# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional
import re


def extract_single_line_metrics(
    text: str,
    metric_types: dict[str, type],
) -> dict:
    """
    Extracts key-value pairs from a text string and casts values to specified types.

    Args:
        metric_types (dict[str, type]): Mapping of keys to their expected Python types.
        text (str): Assumes input text contains key-value pairs in the format "k1: v1 k2: v2,..."

    Returns:
        dict: Extracted key-value pairs with values cast to their respective types, or {} if casting fails.
    """
    pattern = r'(\w+)\s*:\s*([^,\s]+)'

    metrics = {}
    matches = re.findall(pattern, text)

    metric_keys = list(metric_types.keys())

    for key, value in matches:
        if key in metric_keys:
            if metric_types and key in metric_types:
                try:
                    metrics[key] = metric_types[key](value)
                except (ValueError, TypeError):
                    return {}
            else:
                metrics[key] = value

    for key in metric_keys:
        if key not in metrics:
            return {}

    return metrics


def extract_best_line_metrics(
    text: str,
    metric_types: dict[str, type],
    selection_metric: str, 
    lower_is_better=False,
    metrics_at_most: Optional[dict[str, int | float]] = None,
    metrics_at_least: Optional[dict[str, int | float]] = None
) -> dict:
    best_metrics = None
    best_sel_value = None
    for line in text.splitlines():
        is_valid = True
        metrics = extract_single_line_metrics(line, metric_types)
        if not metrics:
            continue

        # Reject if any metrics go below a floor threshold
        if metrics_at_least and any(metrics.get(key, float('inf')) < threshold 
               for key, threshold in metrics_at_least.items()):
            is_valid = False

        # Reject if any metrics exceed a ceiling threshold
        elif metrics_at_most and any(metrics.get(key, float('-inf')) > threshold 
               for key, threshold in metrics_at_most.items()):
            is_valid = False

        # Get the value of the selection metric; if absent, skip.
        sel_val = metrics.get(selection_metric)
        if sel_val is None:
            continue

        metrics['is_valid'] = is_valid
        if best_metrics is None:
            best_metrics, best_sel_val = metrics, sel_val
        else:
            # Only replace if better than current best + is valid under constraints
            if is_valid and ((
                lower_is_better and sel_val < best_sel_val
            ) or (
                not lower_is_better and sel_val > best_sel_val
            )):
                best_metrics, best_sel_val = metrics, sel_val

    if best_metrics is None:
        best_metrics = {}

    if not best_metrics and not metric_types:
        best_metrics['is_valid'] = True

    return best_metrics


def extract_last_line_metrics(
    text: str,
    metric_types: dict[str, type],
):
    metrics = {}
    for line in text.splitlines():
        line_metrics = extract_single_line_metrics(line, metric_types)
        if line_metrics:
            metrics = line_metrics

    if metrics or not metric_types:
        metrics['is_valid'] = True

    return metrics
