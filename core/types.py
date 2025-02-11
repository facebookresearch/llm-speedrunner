from typing import Optional, Union
import dataclasses


Serializable = Union[str, int, float, bool, None, dict[str, "Serializable"], list["Serializable"]]


@dataclasses.dataclass
class ExperimentConfig:
	preamble: str
	idea_instructions: str
	code_instructions: str

	entry_fname: str
	fnames: list[str]

	selection_metric: str
	lower_is_better: bool = False
	metric_types: Optional[dict[str, list[type]]] = None
	metrics_at_least: Optional[dict[str, int | float]] = None
	metrics_at_most: Optional[dict[str, int | float]] = None
	max_retries: int = 3

	eval_fname: Optional[str] = None


@dataclasses.dataclass
class SlurmConfig:
	nodes: int 
	tasks_per_node: int
	gpus_per_node: int 
	cpus_per_task: int
	job_ttl: int
	use_torchrun: bool = False
	job_name: str = 'submitit'
	account: str = 'maui'
	qos: Optional[str] = None
	env_vars: Optional[dict[str, str]] = None
	log_dir='submitit_logs'