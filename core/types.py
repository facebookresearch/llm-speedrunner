from typing import Optional, Union
import dataclasses


Serializable = Union[str, int, float, bool, None, dict[str, "Serializable"], list["Serializable"]]


@dataclasses.dataclass
class ExperimentRecord:
	diffs: list[str]
	metrics: dict[str, Serializable]


@dataclasses.dataclass
class ExperimentHistory:
	records: list[ExperimentRecord]


@dataclasses.dataclass
class ExperimentConfig:
	preamble: str
	job_ttl: int
	max_retries: int = 3
