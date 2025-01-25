from typing import Optional, Union


Serializable = Union[str, int, float, bool, None, dict[str, "Serializable"], list["Serializable"]]


@dataclasses.dataclass
class ExperimentRecord:
	diffs: list[str]
	metrics: dict[str, Serializable]


@dataclass.dataclass
class ExperimentHistory:
	records: list[ExperimentRecord]


@dataclass.dataclass
class ExperimentConfig
	preamble: str
	job_ttl: int
	max_retries: int = 3
