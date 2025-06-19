# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Union
import dataclasses


Serializable = Union[str, int, float, bool, None, dict[str, "Serializable"], list["Serializable"]]


@dataclasses.dataclass
class ExperimentConfig:
	code_instructions: str

	entry_fname: str
	fnames: list[str]

	selection_metric: str
	lower_is_better: bool = False
	metric_types: Optional[dict[str, list[type]]] = None
	metrics_at_least: Optional[dict[str, int | float]] = None
	metrics_at_most: Optional[dict[str, int | float]] = None

	eval_fname: Optional[str] = None
	eval_metric_types: Optional[dict[str, list[type]]] = None
	eval_selection_metric: Optional[str] = None
	eval_lower_is_better: bool = False
	eval_metrics_at_least: Optional[dict[str, int | float]] = None
	eval_metrics_at_most: Optional[dict[str, int | float]] = None
	eval_metrics_private: Optional[list[str]] = None

	task_description: Optional[str] = None
	task_description_file: Optional[str] = None
	preamble: Optional[str] = None
	max_retries: int = 3


@dataclasses.dataclass
class SlurmConfig:
	nodes: int 
	tasks_per_node: int
	gpus_per_node: int 
	cpus_per_task: int
	job_ttl: int
	use_torchrun: bool = False
	use_local_runs: bool = False
	job_name: str = 'submitit'
	account: str = 'maui'
	qos: Optional[str] = None
	env_vars: Optional[dict[str, str]] = None
	log_dir='submitit_logs'