# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import asyncio
import os
import signal
import sys

import hydra
from omegaconf import DictConfig, OmegaConf

from core.types import ExperimentConfig, SlurmConfig
from core.runners.science_runner import ScienceRunner
from core.runners.bon_science_runner import BoNScienceRunner
from utils import fs_utils


async def shutdown(
    loop: asyncio.AbstractEventLoop,
    science_runner: ScienceRunner
):
    print('Shutting down ScienceRunner instance...')
    science_runner.shutdown()
    print('Successfully shut down ScienceRunner instance.')

    tasks = [
        t for t in asyncio.all_tasks(loop) 
        if t is not asyncio.current_task(loop)
    ]
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    loop.stop()


async def main_async(cfg: DictConfig):
    # Set the HYDRA_FULL_ERROR environment variable
    os.environ['HYDRA_FULL_ERROR'] = '1'
    # Load existing config if it exists (e.g. reentering a preempted run)
    ws_root_path = fs_utils.expand_path(cfg.workspace_args.root_path)
    cfg_path = os.path.join(ws_root_path, 'config.yaml')
    if os.path.exists(cfg_path):
        existing_cfg = OmegaConf.load(cfg_path)
        existing_cfg.workspace_args.root_path = cfg.workspace_args.root_path
        if cfg.n_iterations > existing_cfg.n_iterations:
            existing_cfg.n_iterations = cfg.n_iterations  # Allow overriding n_iterations
        cfg = existing_cfg
        print(f'Using config for existing run at {cfg_path}.')

    science_runner = hydra.utils.instantiate(cfg.science_runner_args)

    with open(cfg_path, "w") as f:
        OmegaConf.save(cfg, f)

    # Register signal handlers
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(
            sig, lambda: asyncio.create_task(
                shutdown(loop, science_runner)
            )
        ) 

    try:
        await science_runner.run(n_iterations=cfg.n_iterations)
    except asyncio.exceptions.CancelledError:
        print('Preparing to shut down scientist...')


@hydra.main(config_path="config", config_name="default.yaml", version_base="1.1")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    asyncio.run(main_async(cfg))


if __name__ == '__main__':
    main()