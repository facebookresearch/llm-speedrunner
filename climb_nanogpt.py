from typing import Optional, Type

import datetime
import dataclasses
import logging
import os
import subprocess

from scientist.llm_client import LLMClient
import scientist.prompts as prompts
import scientist.slurm_utils as slurm_utils


Serializable = Union[str, int, float, bool, None, Dict[str, "Serializable"], List["Serializable"]]


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
	workspace: Workspace
	scientist: Agent
	job_ttl: int
	max_retries: int = 3


class Workspace:
	"""Global workspace for the scientist. Tracks relevant artifacts.

	Workspace contains:
		- Directory reference to project files
		- Chain of experiment diffs from base project
		- Evaluation metrics per experiment
	"""

	def __init__(self, root_dir: str, track_history=True):
		self.root_dir: str = root_dir

		if track_history:
			self._exp_history: ExperimentHistory = ExperimentHistory(records=[])
		else:
			self._exp_history = None

		# Find all version directories
		version_dirs = []
		pattern = re.compile(r'^version_\d+$')  # Matches 'version_' followed by an integer
		for dirname in os.listdir(root_path):
		    abs_dir_path = self._get_abs_path(dirname)
		    if os.path.isdir(full_path) and pattern.match(dirname):
		        version_dirs.append(abs_dir_path)

		if not version_dirs:
			version_dirs.append(self._get_abs_path('version_0'))

		self.version_dir: str = version_dirs[-1]

	def _get_abs_path(self, path: str):
		return os.path.join(self.root_dir, path)

	def save_to_file(self, text: str, path: str):
		"""Save text content to a file path in root_dir."""
		with open(self._get_abs_path(path), 'w') as fout:
			fout.write(str)

	def create_version(self):
		self.n_versions += 1
		os.makedirs(self._get_abs_path(f'version_{self.n_versions}'), exist_ok=True)

	def obs(self, parts: list[str]):
		"""Observe full workspace, or part of it.
		"""
		with open('train_gpt.py', 'w') as file_train_py:
			train_py = file_train_py.read()

		return train_py

	def obs_dir(self, paths: Optional[list[str]] = None):
		"""Observe contents of working directory.
	
		Arguments
			paths: If set, only observe these paths; otherwise, 
			observe everything in the root_dir.

		Returns
			A dictionary mapping paths to contents.
		"""
		pass

	def exec_cmd(self, cmd: str):
		return subprocess.run(cmd, shell=True, capture_output=True, text=True)

	def obs_history(
		self, 
		max_len: Optional[int] = None, 
		as_diffs=True, 
		valid_only=True
	) -> list[ExperimentRecord] | None:
		"""Observe history of experiments and results.

		Arguments
			max_len: Only return this many of most recent records.
			as_diffs: If True, return as a chain of diffs from first returned record.


		Returns
			A list of the max_len most recent experiment records.
		"""
		if track_history:
			return self._exp_history[-max_len:]
		else:
			return None

	def save_to_history(self, record: ExperimentRecord):
		"""Save a new experiment record to experiment history."""
		if not track_history:
			return

		self._exp_history.append(record)



class Agent:
	def __init__(self, model="gpt-4o", system_prompt: str):
		LLMClient(model=model, system_prompt=system_prompt)

	def act(self, instruction: str, validator: Optional[Callable[str, bool]], max_retries=1) -> str:
		response = LLMClient.generate(instruction)

		if validator and not validator(response):
			n_retries = 0
			while n_retries < max_retries:
				response = LLMClient.generate(instruction)

				n_retries += 1

				if validator(response):
					return response
		else:
			return response

		raise ValueError(f'Malformed response after {max_retries} attempts.')


class ExperimentRunner:
	def __init__(self, config: ExperimentConfig)
		self.preamble = config.preamble
		self.max_retries = config.max_retries
		self.workspace = config.Workspace
		self.scientist = config.scientist
		self.job_ttl = config.job_ttl

	def get_instruction(self, instruction: str) -> str:
		return '\n'.join([self.preamble, instruction])

	def run_scientist(
		self, 
		instruction: str, 
		validator: Optional[Callable[str, bool]] = None, 
		max_retries=Optional[int] = None
	) -> str:
		if max_retries is None:
			max_retries = self.max_retries

		try:
			response = self.scientist.act(
				self.get_instruction(instruction),
				validator=lambda x: validate_json(x, dict('hypothesis': str)),
				max_retries=max_retries
			)
		except:
			logging.info("Bad response. Terminating experiment prematurely.")

		return response

	def run_exp(self):
		raise NotImplementedError()

	def run(self, n_iterations=1):
		for i in range(n_iterations):
			self.workspace.create_version()
			run_exp()


# Validators
def validate_json(x: str, type_dict: Optional[dict[str, Type]] = None) -> bool:
	data = None

	try:
		json.loads(x)
	except:
		return False

	if type_dict:
		for k,v in type_dict.items():
			if not k in data or not isinstance(data[k], v):
				return False

	return True


class NanoGPTClimber(ExperimentRunner):
	def init_workspace(self):
		# @todo: Copy files into workspace
		pass

	def run_exp(self):
		# Observe dir + history, if enabled
		code = self.workspace.obs_dir('train_gpt.py')

		# Request next hypothesis
		hypothesis = self.run_scientist(
			prompt.NANOGPT_TASK_GENERATE_HYPOTHESIS.format(code=code)
			validator=lambda x: validate_json(x, dict('hypothesis': str)),
		)['hypothesis']

		# Implement hypothesis
		updated_code = self.run_scientist(
			prompt.NANOGPT_TASK_IMPLEMENT_HYPOTHESIS.format(code=code, hypothesis=hypothesis)
		)

		# Save code to workspace_dir, @todo: add linting
		self.workspace.save_to_file(updated_code, 'train_gpt.py')

		# @todo: Launch experiment on slurm cluster + track jobid
		job_id = slurm_utils.launch_job(
			command="train_gpt.py", 
			n_nodes=1, 
			gpus_per_node=8, 
			cpus_per_task=96, 
			tasks_per_node=1, 
			timeout_min=self.job_ttl,
			job_name='maui_climber',
			account='maui',
			qos='maui_high',
		)

		# When time-limit exceeded, check for experiment results
		sleep(self.job_ttl)
		job_out = slurm_utils.get_job_status(job_id)
		
		# Parse experiment results @todo: read results from logs
		fitness = 0
		if job_out:
			try:
				fitness = 1
			except:
				pass

		# Record experiment results in history
		# - Compute diff from previous code
		# - Store diffs in code

		exp_metrics = dict(
			val_loss=val_loss,
			train_loss=train_loss,
			walltime=walltime
		)
		pass


def main():
	# Create scientist agent
	preamble = prompts.NANOGPT_CLIMBER_TASK_PREAMBLE
	scientist = Agent(model='qwen-r1-32b', system_prompt=prompts.SCIENTIST_SYSTEM_PROMPT)

	root_dir = 'nanogpt_workspaces/' + datetime.now().strftime('%Y%m%d_%H%M%S_%f')
	workspace = Workspace(root_dir=root_dir)  # @todo: Should generate a new experiment directory

	exp_config = ExperimentConfig(
		preamble=preamble,
		workspace=workspace, 
		scientist=scientist,
		job_ttl=10*60  # seconds
	)

	climber = NanoGPTClimber(workspace, scientist)

	climber.run(n_iterations=10)


if __name__ == '__main__':
	main()