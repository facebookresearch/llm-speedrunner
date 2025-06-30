# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, Optional, Union
import dataclasses
import json
import logging
import os

from utils import metrics_utils
from utils import slurm_utils
from utils import str_utils
from utils import fs_utils
from core.types import ExperimentConfig, SlurmConfig
from core.agent import Agent
from core.coders.base import Coder
from core.ideators.base import Ideator
from core.workspace import Workspace, VersionInfo
from core.prompts import analysis_prompts
from core import validators


class ScienceRunner:
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
        max_n_nodes=20
    ):
        self.preamble = config.preamble
        
        # Load task description file in template dir if provided
        task_description = config.task_description
        if config.task_description_file:
            task_description_path = workspace.resolve_path(
                config.task_description_file,
                version='0'
            )

            if os.path.exists(task_description_path):
                with open(task_description_path, 'r') as f:
                    task_description_loaded = f.read().strip()
                    if task_description:
                        task_description = task_description_loaded + '\n\n' + task_description
                    else:
                        task_description = task_description_loaded

        assert task_description, 'Must provide a valid task description.'
        self.task_description = task_description

        self.code_instructions = config.code_instructions
        self.fnames = config.fnames

        self.entry_fname = config.entry_fname
        self.eval_fname = config.eval_fname
        self.slurm_config = slurm_config
        self.eval_slurm_config = eval_slurm_config

        self.workspace = workspace

        self.selection_metric = config.selection_metric
        self.lower_is_better = config.lower_is_better
        self.metric_types = {
            k: str_utils.basic_type_name_to_type(v) 
            for k,v in config.metric_types.items()
        }
        self.metrics_at_least = config.metrics_at_least
        self.metrics_at_most = config.metrics_at_most

        if config.eval_metric_types:
            self.eval_metric_types = {
                k: str_utils.basic_type_name_to_type(v) 
                for k,v in config.eval_metric_types.items()
            }
        else:
            self.eval_metric_types = None
        self.eval_selection_metric = config.eval_selection_metric
        self.eval_lower_is_better = config.eval_lower_is_better
        self.eval_metrics_at_least = config.eval_metrics_at_least
        self.eval_metrics_at_most = config.eval_metrics_at_most
        self.eval_metrics_private = config.eval_metrics_private or []

        self.max_retries = max_retries
        self.max_log_len = max_log_len
        self.max_n_nodes = max_n_nodes

        # Agents
        self.assistant = assistant
        self.ideator = ideator
        self.coder = coder

    def get_instruction(self, instruction: str) -> str:
        return '\n'.join([self.preamble, instruction])

    def extract_metrics(
        self,
        version: str,
        logs: str,
        is_eval=False
    ):
        version_info = self.workspace.version_infos[version]
        metrics = {}

        if is_eval:
            selection_metric = self.eval_selection_metric
            lower_is_better = self.eval_lower_is_better
            metric_types = self.eval_metric_types
            metrics_at_least = self.eval_metrics_at_least
            metrics_at_most = self.eval_metrics_at_most
        else:
            selection_metric = self.selection_metric
            lower_is_better = self.lower_is_better
            metric_types = self.metric_types
            metrics_at_least = self.metrics_at_least
            metrics_at_most = self.metrics_at_most

        if selection_metric:
            metrics = metrics_utils.extract_best_line_metrics(
                logs, 
                metric_types=metric_types,
                selection_metric=selection_metric,
                lower_is_better=lower_is_better,
                metrics_at_least=metrics_at_least,
                metrics_at_most=metrics_at_most
            )
        else:
            metrics = metrics_utils.extract_last_line_metrics(
                logs,
                metric_types=metric_types
            )

        # If no regex match on results
        if not metrics:
            metric_types_str = json.dumps({k: v.__name__ for k, v in metric_types.items()})

            try:
                metrics_response = self.assistant.act(
                    analysis_prompts.PARSE_METRICS_FROM_LOGS.format(
                        logs=logs, metric_types=metric_types_str
                    ),
                    validator=lambda x: validators.validate_json(x, metric_types),
                    max_retries=self.max_retries
                )
            except:
                metrics_response = {}
            print(f'metrics_response:\n{metrics_response}')
            if metrics_response:
                metrics = json.loads(metrics_response)
                if not any([v is None for _, v in metrics.items()]):
                    metrics['is_valid'] = True
                else:
                    metrics['is_valid'] = False

        # Reject if any metrics go below a floor threshold
        if self.metrics_at_least and any(metrics.get(key, float('inf')) < threshold 
               for key, threshold in self.metrics_at_least.items()):
            metrics['is_valid'] = False

        # Reject if any metrics exceed a ceiling threshold
        if self.metrics_at_most and any(metrics.get(key, float('-inf')) > threshold 
               for key, threshold in self.metrics_at_most.items()):
            metrics['is_valid'] = False

        # In the worst case, default to empty metrics with previous keys
        if not metrics:
            summary = json.loads(
                self.workspace.view(
                    'results.json', 
                    version=version_info.parent_version,
                    no_filename_headers=True).strip()
            )
            metrics = {k: None for k, _ in summary.get('metrics', {}).items()}
            metrics['is_valid'] = False

        return metrics

    def set_results_for_version(
        self,
        version: str,
        job_results: slurm_utils.JobResult,
        eval_job_results: Optional[slurm_utils.JobResult] = None
    ):
        version_info = self.workspace.version_infos[version]

        # Summarize the main job results
        log_out = job_results.log_out[0][-self.max_log_len:]
        log_err = job_results.log_err[0][-self.max_log_len:]
        outcome_summary = self.assistant.act(
            analysis_prompts.SUMMARIZE_LOGS_PROMPT.format(
                goal=self.task_description,
                log_out=log_out,
                log_err=log_err
            ),
            max_retries=self.max_retries
        )
        print(f'OUTCOME SUMMARY:\n{outcome_summary}')

        # Extract metrics from job logs
        metrics = {}
        if self.selection_metric and log_out:
            metrics = self.extract_metrics(version, log_out)

        eval_metrics = {}
        if eval_job_results is not None:
            eval_log_out = eval_job_results.log_out[0][-self.max_log_len:]
            eval_metrics = self.extract_metrics(version, eval_log_out, is_eval=True)

        # Move private metrics into a separate private_metrics dict:
        metrics.update(eval_metrics)

        private_metrics = {
            k: v for k, v in metrics.items()
            if k != 'is_valid' and (k in self.eval_metrics_private or ('*' in self.eval_metrics_private and k in self.eval_metric_types))
        }
        for k in private_metrics:
            del metrics[k]
        metrics['private'] = private_metrics

        # Update meta based on whether job failed or failed to produce usable metrics
        if job_results.status == slurm_utils.JobStatus.FAILED or not metrics['is_valid']:  # Update meta
            self.workspace.mark_as_buggy_from_version(
                version=version, 
                from_version=version_info.parent_version
            )

        metadata = str_utils.get_serializable_dict_subset(job_results.metadata)
        job_results = {
            'status': job_results.status.value,
            'metrics': metrics,
            **metadata,
            'outcome_summary': outcome_summary
        }

        self.workspace.save_to_file(
            json.dumps(job_results, indent=4), 'results.json', version=version
        )

    async def run(self, n_iterations=1):
        raise NotImplementedError()

    def shutdown(self):
        slurm_utils.JobObserver.shared.cancel()
        pending_version_infos = self.workspace.get_pending_versions()
        for info in pending_version_infos:
            self.workspace.delete_version(info.version)

