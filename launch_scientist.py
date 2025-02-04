import argparse
import asyncio

import hydra
from omegaconf import DictConfig

from core.types import ExperimentConfig, SlurmConfig
from core.runners.bon_science_runner import BoNScienceRunner


async def main_async(cfg: DictConfig):
	science_runner = hydra.utils.instantiate(cfg.science_runner_args)

	await science_runner.run(n_iterations=cfg.n_iterations)


@hydra.main(config_path="config", config_name="default.yaml", version_base="1.1")
def main(cfg: DictConfig):
	print(cfg)
	asyncio.run(main_async(cfg))


if __name__ == '__main__':
	main()
