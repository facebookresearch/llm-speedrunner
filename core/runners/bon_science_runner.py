# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, Optional
import asyncio
import dataclasses
import logging
import submitit

import numpy as np

from utils import slurm_utils
from core.types import ExperimentConfig, SlurmConfig
from core.agent import Agent
from core.coders.base import Coder
from core.ideators.base import Ideator
from core.knowledge import KnowledgeStore
from core.runners.science_runner import ScienceRunner
from core.workspace import Workspace, VersionInfo

import random
import json


class BoNScienceRunner(ScienceRunner):
    """Science loop that evaluates N hypotheses, and continues with the BoN."""
    def __init__(
        self, 
        config: ExperimentConfig, 
        workspace: Workspace, 
        assistant: Agent,
        ideator: Ideator,
        coder: Coder,
        slurm_config: SlurmConfig,
        eval_slurm_config: Optional[SlurmConfig] = None,
        max_retries=3,
        max_log_len=30_000,
        max_n_nodes=20,
        n_hypotheses=1,
        n_initial_hypotheses=1,
        debug_prob=0.0,
        max_bug_depth: Optional[int] = 3,
        knowledge_src_paths: Optional[list[str]] = None,
        knowledge_pass_to_coder: bool = False
    ):
        super().__init__(
            config=config,
            workspace=workspace,
            assistant=assistant,
            ideator=ideator,
            coder=coder,
            slurm_config=slurm_config,
            eval_slurm_config=eval_slurm_config,
            max_retries=max_retries,
            max_log_len=max_log_len,
            max_n_nodes=max_n_nodes
        )

        self.n_hypotheses = n_hypotheses
        self.n_initial_hypotheses = n_initial_hypotheses
        self.debug_prob = debug_prob
        self.max_bug_depth = max_bug_depth

        self.knowledge = KnowledgeStore(src_paths=knowledge_src_paths)
        self.knowledge_pass_to_coder = knowledge_pass_to_coder

    def _job_callback(
        self,
        version: str,
        job_results: slurm_utils.JobResult,
        eval_job_results: Optional[slurm_utils.JobResult] = None
    ):
        self.set_results_for_version(version, job_results, eval_job_results)

        self.assistant.flush_logs(
            self.workspace.resolve_path('assistant_history.jsonl', 
            version=version)
        )
        self.ideator.flush_logs(
            self.workspace.resolve_path('ideator_history.jsonl',
            version=version)
        )
        self.ideator.flush_logs(
            self.workspace.resolve_path('coder_history.jsonl',
            version=version)
        )

    async def _run_exp(
        self,
        version_info: VersionInfo, 
        metadata: Optional[dict[str, str | int | float, bool]] = None
    ):
        version = version_info.version
        history_from_version = version_info.stable_ancestor_version
        bug_history = self.workspace.view_history(
            from_version=history_from_version,
            max_len=3,
            incl_good_versions=False,
            incl_buggy_versions=True,
            incl_ancestors=False,
            incl_descendents=True,
            descendent_depth=3,
            as_string=True
        )

        # Keep main thread unblocked
        loop = asyncio.get_running_loop()
        coder_out = await loop.run_in_executor(
            None, 
            lambda: self.coder.code(
                task_description=self.task_description,
                instruction=self.code_instructions,
                ideas=metadata.get('hypothesis'),
                fnames=self.fnames,
                workspace=self.workspace,
                version=version,
                bug_history=bug_history,
                knowledge=None if not self.knowledge_pass_to_coder else self.knowledge.search(as_string=True),
                max_retries=self.max_retries
            )
        )
        print(f'Coder out:\n{coder_out}')

        # Send experiment to slurm
        if self.slurm_config.use_torchrun:
            command = self.entry_fname
        else:
            command = f'python {self.entry_fname}'

        with submitit.helpers.clean_env():
            job = slurm_utils.submit_job(
                command=command, 
                working_dir=self.workspace.resolve_path(version=version),
                **dataclasses.asdict(self.slurm_config)
            )

            # Monitor experiment status and bookkeep final outcome
            callback = None
            if self.eval_fname is None:
                callback = lambda res: self._job_callback(version, res)
            else:  # We save the job result to pass into the eval callback
                def record_job_result(res: slurm_utils.JobResult):
                    metadata[version] = res
                callback = lambda res: record_job_result(res)

            slurm_utils.JobObserver.shared.observe(
                job=job,
                metadata=metadata,
                callback=callback,
            )

    async def _run_eval(
        self, 
        version_info: VersionInfo,
        metadata: Optional[dict[str, str | int | float, bool]] = None
    ):
        version = version_info.version
        if self.eval_fname is not None:
            eval_slurm_config = self.eval_slurm_config
            if not eval_slurm_config:
                eval_slurm_config = self.slurm_config

            if eval_slurm_config.use_torchrun:
                command = self.eval_fname
            else:
                command = f'python {self.eval_fname}'

            eval_job = slurm_utils.submit_job(
                command=f"python {self.eval_fname}", 
                working_dir=self.workspace.resolve_path(version=version),
                **dataclasses.asdict(eval_slurm_config)
            )

            job_result = metadata.get(version)
            slurm_utils.JobObserver.shared.observe(
                job=eval_job,
                metadata=metadata,
                callback=lambda res: self._job_callback(version, job_result, res)
            )

    def select_next_open_version(self, current_versions: list[str]):
        # If debug, then select a buggy leaf version to debug
        buggy_versions = self.workspace.get_buggy_versions(
            is_leaf=True, max_bug_depth=self.max_bug_depth
        )
        if len(buggy_versions) and np.random.rand() < self.debug_prob:
            open_version = np.random.choice(buggy_versions).version
        else: # Otherwise set open set to the top-1 version
            open_version = self.workspace.get_top_k_versions(
                selection_metric=self.eval_selection_metric or self.selection_metric,
                from_versions=current_versions,
                lower_is_better=self.lower_is_better,
                k=1
            )[0].version

        return open_version

    async def run(self, n_iterations=1):
        # Offset iteration and hypotheses indexing if we are re-entering
        # a preempted run.
        completed_version_infos = self.workspace.get_completed_versions()
        start_iter_idx = len(completed_version_infos)

        if start_iter_idx < self.n_initial_hypotheses:
            n_initial_hypotheses = self.n_initial_hypotheses - start_iter_idx
            start_iter_idx = 0

        # Get starting open version (None defaults to template)
        if start_iter_idx > 0:
            open_version = self.select_next_open_version(
                [info.version for info in completed_version_infos]
            )
        else:
            open_version = None

        for i in range(start_iter_idx, n_iterations):
            # Check if we've reached the maximum number of nodes
            if self.workspace.n_versions - 1 >= self.max_n_nodes:
                logging.info(f"Maximum number of nodes ({self.max_n_nodes}) reached. Stopping runner.")
                break
                
            # Request next hypotheses
            relevant_history = None
            is_debugging = False
            if open_version is not None:
                open_version_info = self.workspace.get_version_info(open_version)
                is_debugging = open_version_info.bug_depth > 0
                history_from_version = open_version_info.stable_ancestor_version
                relevant_history = self.workspace.view_history(
                    from_version=history_from_version,
                    max_len=3,
                    incl_good_versions=True,
                    incl_buggy_versions=True,
                    incl_ancestors=True,
                    incl_descendents=True,
                    ancestor_depth=3,
                    descendent_depth=1,
                    as_string=True
                )

            # By default, we branch n_hypotheses experiments per iteration, but
            # keep the branch factor to be 1 when debugging a previous version.
            # First iteration's branch factor can be set separately to mimic AIDE.
            n_hypotheses = self.n_hypotheses
            if is_debugging:
                n_hypotheses = 1
            elif i == 0:
                n_hypotheses = n_initial_hypotheses

            loop = asyncio.get_running_loop()
            hypotheses = []
            for _ in range(n_hypotheses):
                # Keep main thread unblocked
                hypothesis, _ = await loop.run_in_executor(
                    None, 
                    lambda: self.ideator.ideate(
                        task_description=self.task_description,
                        fnames=self.fnames,
                        workspace=self.workspace,
                        version='0' if not open_version else open_version,
                        ignore_ideas=hypotheses,  # Avoid duplicating previous ideas
                        history=relevant_history,
                        knowledge=self.knowledge.search(as_string=True),
                        max_retries=self.max_retries
                    )
                )
                hypotheses.append(hypothesis)

            current_versions = ['0']
            if open_version is not None and open_version not in current_versions:
                current_versions.append(open_version)  # Always consider best so far

            version2metadata = {}
            for hyp_idx, hypothesis in enumerate(hypotheses):
                print(f'Hypothesis:\n{hypothesis}')

                # Check if we've reached the maximum number of nodes
                if self.workspace.n_versions - 1 >= self.max_n_nodes:
                    logging.info(f"Maximum number of nodes ({self.max_n_nodes}) reached. Skipping remaining hypotheses.")
                    break

                # Create a new workspace version for each experiment
                if i == 0:
                    # All first generation hypotheses branch from template
                    version = self.workspace.create_version(from_version='0')
                else:
                    # All other hypotheses branch from best version so far
                    prev_version = open_version
                    version = self.workspace.create_version(
                        from_version=prev_version
                    )
                
                # Log node count
                logging.info(f"Created node {self.workspace.n_versions - 1}/{self.max_n_nodes}")

                current_versions.append(str(version))
                version2metadata[version] = {
                    'hypothesis': hypothesis
                }

                # Schedule experiments for all hypotheses
                version_info = self.workspace.get_version_info(version)

                await self._run_exp(
                    version_info=version_info,
                    metadata=version2metadata[version]
                )

            # Wait for all experiments to finish
            await slurm_utils.JobObserver.shared.wait()

            if self.eval_fname is not None:
                print('EVALUATING via', self.eval_fname)
                for version in current_versions:
                    if version == '0' or version == open_version:
                        continue

                    version_info = self.workspace.get_version_info(version)
                    metadata = version2metadata[version]
                    await self._run_eval(
                        version_info=version_info,
                        metadata=metadata
                    )

                print('Waiting for evals to finish')
                await slurm_utils.JobObserver.shared.wait()

            # Check if we've reached the maximum number of nodes before selecting next version
            if self.workspace.n_versions - 1 >= self.max_n_nodes:
                logging.info(f"Maximum number of nodes ({self.max_n_nodes}) reached. Stopping runner.")
                break
                
            open_version = self.select_next_open_version(current_versions)
