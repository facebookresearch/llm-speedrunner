# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from scientist.utils import metrics_utils


METRIC_TYPES = {"acc": float, "loss": float, "epoch": int}

MOCK_LOGS = """
step: 1 acc: 0.95, loss: 0.12 epoch: 1 
step: 2 acc: 0.96, loss: 0.11 epoch: 2 
step: 3 acc: 0.98, loss: 0.09 epoch: 3 
step: 4 acc: 0.97, loss: 0.10 epoch: 4 
"""


def test_extract_single_line_metrics():
	text = "step: 1 acc: 0.95, loss: 0.12 epoch: 10 "

	metrics = metrics_utils.extract_single_line_metrics(text, METRIC_TYPES)

	assert metrics == {'acc': 0.95, 'loss': 0.12, 'epoch': 10}


def test_extract_single_line_metrics_bad_type():
	text = "step: 1 acc: 0.95, loss: 0.12 epoch: test "

	metrics = metrics_utils.extract_single_line_metrics(text, METRIC_TYPES)

	assert metrics == {}


def test_extract_best_line_metrics_higher_is_better():
	text = MOCK_LOGS

	metrics = metrics_utils.extract_best_line_metrics(
		text, 
		metric_types=METRIC_TYPES,
		selection_metric='acc',
	)

	assert metrics == {'acc': 0.98, 'loss': 0.09, 'epoch': 3, 'is_valid': True}


def test_extract_best_line_metrics_lower_is_better():
	text = MOCK_LOGS
	metrics = metrics_utils.extract_best_line_metrics(
		text, 
		metric_types=METRIC_TYPES,
		selection_metric='loss',
		lower_is_better=True
	)

	assert metrics == {'acc': 0.98, 'loss': 0.09, 'epoch': 3, 'is_valid': True}


def test_extract_best_line_metrics_lower_is_better_at_most():
	text = MOCK_LOGS
	metrics = metrics_utils.extract_best_line_metrics(
		text, 
		metric_types=METRIC_TYPES,
		selection_metric='loss',
		lower_is_better=True,
	)

	assert metrics == {'acc': 0.98, 'loss': 0.09, 'epoch': 3, 'is_valid': True}


def test_extract_best_line_metrics_lower_is_better_at_least():
	text = MOCK_LOGS
	metrics = metrics_utils.extract_best_line_metrics(
		text, 
		metric_types=METRIC_TYPES,
		selection_metric='loss',
		lower_is_better=True,
		metrics_at_least={'epoch': 4}
	)

	assert metrics == {'acc': 0.97, 'loss': 0.10, 'epoch': 4, 'is_valid': True}


def test_extract_best_line_metrics_lower_is_better_at_most():
	text = MOCK_LOGS
	metrics = metrics_utils.extract_best_line_metrics(
		text, 
		metric_types=METRIC_TYPES,
		selection_metric='loss',
		lower_is_better=True,
		metrics_at_most={'epoch': 2}
	)

	assert metrics == {'acc': 0.96, 'loss': 0.11, 'epoch': 2, 'is_valid': True}


def test_extract_best_line_metrics_lower_is_better_mixed_thresholds():
	text = MOCK_LOGS
	metrics = metrics_utils.extract_best_line_metrics(
		text, 
		metric_types=METRIC_TYPES,
		selection_metric='loss',
		lower_is_better=True,
		metrics_at_most={'epoch': 2},
		metrics_at_least={'loss': 0.12}
	)

	assert metrics == {'acc': 0.95, 'loss': 0.12, 'epoch': 1, 'is_valid': True}


def test_extract_best_line_metrics_lower_is_better_no_match():
	text = MOCK_LOGS
	metrics = metrics_utils.extract_best_line_metrics(
		text, 
		metric_types=METRIC_TYPES,
		selection_metric='loss',
		lower_is_better=True,
		metrics_at_most={'epoch': 2},
		metrics_at_least={'epoch': 3}
	)

	assert metrics == {'acc': 0.95, 'loss': 0.12, 'epoch': 1, 'is_valid': False}

