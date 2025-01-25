

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

	def run(self, n_iterations=1):
		raise NotImplementedError()
