# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Optional, Tuple

import asyncio
import dataclasses
import os
import pickle
import re
import submitit
import subprocess

from utils import fs_utils


class JobStatus(Enum):
    COMPLETED = 'COMPLETED'
    PREEMPTED = 'PREEMPTED'
    FAILED = 'FAILED'
    CANCELLED = 'CANCELLED'
    UNKNOWN = 'UNKNOWN'


@dataclasses.dataclass
class JobResult:
    """Aggregate information about the finished job."""
    job_id: str
    metadata: dict[str, str | int | float, bool]
    status: JobStatus
    log_out: list[str]
    log_err: list[str]

    # @todo(msj): Should track time_to_start and runtime


class JobObserver:
    """Manages a pool of asyncio tasks for observing submitit job status."""
    shared = None  # Singleton

    def __init__(self):
        # We keep a list of tasks created by 'observe'
        self._observing_tasks: list[asyncio.Task] = []
        self._observed_jobs: dict[str, submitit.Job] = {}

    def observe(
        self,
        job: submitit.Job,
        metadata: Optional[dict[str, str | int | float, bool]] = None,
        callback: Optional[Callable[[JobResult], None]] = None,
        focus_rank: Optional[int] = None,
        poll_interval: int = 10,
    ) -> None:
        """
        Observe the status of a submitit job, and execute a callback when finished.

        Args:
            job (submitit.Job): The Submitit job to watch.
            metadata (dict): Some data you want to associate with this job.
            callback (Callable[[JobResult], None]): Called after job finishes with the JobResult.
            focus_rank (Optional[int]): If set, only fetch logs from that subtask index.
            poll_interval (int): How often (seconds) to poll the job status.
        """
        task = asyncio.create_task(
            self._observe_job(
                job=job,
                poll_interval=poll_interval,
                focus_rank=focus_rank,
                callback=callback,
                metadata=metadata
            )
        )
        self._observing_tasks.append(task)
        self._observed_jobs[job.job_id] = job

    async def _observe_job(
        self,
        job: submitit.Job,
        poll_interval=30,
        focus_rank: Optional[int] = None,
        callback: Optional[Callable[[JobResult], None]] = None,
        metadata: Optional[dict[str, str | int | float, bool]] = None,
    ) -> None:
        """
        Loop that periodically checks the job until it's done,
        then calls the user-specified callback with a JobResult.
        """
        while not job.done():
            print('polling job', job, job.done(), job.state.upper(), flush=True)
            await asyncio.sleep(poll_interval)

        # Once we exit the loop, the job is done from Submitit's perspective,
        # meaning it's not in [PENDING, RUNNING, REQUEUED, ...].
        slurm_state = job.state.upper() if job.state else 'UNKNOWN'
        status = self._map_slurm_state_to_job_status(slurm_state, job)

        # Gather logs (stdout & stderr). Focus rank is optional.
        log_out, log_err = self._get_logs(job, focus_rank)

        # Create the result object
        result = JobResult(
            job_id=job.job_id,
            metadata=metadata,
            status=status,
            log_out=log_out,
            log_err=log_err,
        )

        if job.job_id in self._observed_jobs:
            del self._observed_jobs[job.job_id]

        if callback is not None:
            callback(result)

    async def wait(self) -> None:
        """Returns only when all observed jobs and their callbacks are complete."""
        if self._observing_tasks:
            await asyncio.gather(*self._observing_tasks)

    def cancel(self) -> None:
        """Cancel all pending jobs."""
        if len(self._observed_jobs) > 0:
            for _, job in self._observed_jobs.items():
                if job._cancel_command != 'dummy':
                    job.cancel()

    def _map_slurm_state_to_job_status(
        self,
        slurm_state: str,
        job: submitit.Job
    ) -> JobStatus:
        """
        Convert a Slurm or Submitit job state to simpler JobStatus states
        """
        if slurm_state == 'COMPLETED':
            return JobStatus.COMPLETED

        elif slurm_state == 'RUNNING':
            return JobStatus.PREEMPTED

        elif slurm_state in ('CANCELLED'):
            return JobStatus.CANCELLED

        elif job.exception() is not None:
            return JobStatus.FAILED

        elif slurm_state in ('FAILED', 'NODE_FAIL', 'TIMEOUT'):
            return JobStatus.FAILED

        return JobStatus.UNKNOWN

    def _get_logs(
        self,
        job: submitit.Job,
        focus_rank: Optional[int]
    ) -> Tuple[list[str], list[str]]:
        """
        Gathers job stdout/stderr logs as lists of strings.

        If focus_rank is given, only logs for that subtask index.
        Otherwise, all tasks in an array job (or the single task in a normal job).
        """
        # If a particular rank is requested, only retrieve logs from that subtask
        if focus_rank is not None:
            subjob = job.task(focus_rank)
            out = subjob.stdout() or ''
            err = subjob.stderr() or ''
            return [out], [err]

        # If the job has no subtasks (single job), gather from itself
        if not job._sub_jobs:
            return [job.stdout() or ''], [job.stderr() or '']

        # Otherwise, gather from each subtask in the array job
        outs, errs = [], []
        for sub in job._sub_jobs:
            out = sub.stdout() or ''
            err = sub.stderr() or ''
            outs.append(out)
            errs.append(err)
        return outs, errs

JobObserver.shared = JobObserver()  # Initialize the singleton


def submit_job(
    command: str,
    nodes: int,
    cpus_per_task: int,
    gpus_per_node: int,
    tasks_per_node: int,
    job_ttl: int,
    job_name: str,
    account: str,
    qos: Optional[str] = None,
    working_dir: str = '.',
    log_dir='submitit_logs',
    bwrap=True,
    env_vars: Optional[dict[str, str]] = None,
    use_torchrun=False,
    use_local_runs=False,
):
    """
    Launches a SLURM job using submitit with the specified arguments.

    Args:
        entry (str): Path to the Python file that serves as the job entry point.
        num_nodes (int): Number of nodes for the SLURM job.
        cpus_per_task (int): Number of CPUs per task.
        gpus_per_node (int): Number of GPUs per node.
        tasks_per_node (int): Number of tasks per node.
        timeout_min (int): Timeout for the job in minutes.
        job_name (str): Name of the SLURM job.
        account (str): SLURM account to use for the job.
        qos (str): Quality of Service (QoS) for the job.

    Returns:
        str: Job ID of the submitted SLURM job.
    """
    # Local runs
    if use_local_runs:
        # Simulate job and return it
        return simulate_local_submitit_job(
            command=command,
            working_dir=working_dir,
            log_dir=log_dir,
            env_vars=env_vars
        )

    # Create a Submitit executor
    executor = submitit.AutoExecutor(folder=log_dir)

    # SLURM configuration
    executor.update_parameters(
        nodes=nodes,
        cpus_per_task=cpus_per_task,
        gpus_per_node=gpus_per_node,
        tasks_per_node=tasks_per_node,
        timeout_min=job_ttl,
        job_name=job_name,
        slurm_account=account,
        slurm_qos=qos,
        slurm_additional_parameters={
            'chdir': working_dir,
            'gres': f'gpu:{gpus_per_node}',  # Request GPUs
        },
    )

    if use_torchrun:
        def job_function():
            import os
            import socket
            import subprocess

            def find_free_port():
                """Find an available port on the system."""
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(('', 0))  # Bind to any available port
                    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    return s.getsockname()[1]  # Return the port number

            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

            # Set up SLURM environment variables
            master_addr = subprocess.getoutput('scontrol show hostname $SLURM_NODELIST | head -n 1').strip()
            os.environ['MASTER_ADDR'] = master_addr

            # Dynamically set the master port
            master_port = find_free_port()
            os.environ['MASTER_PORT'] = str(master_port)

            os.environ['WORLD_SIZE'] = str(int(nodes * tasks_per_node))  # Total processes
            os.environ['RANK'] = os.environ.get('SLURM_PROCID', '0')  # Rank assigned by SLURM

            # Debugging logs for distributed setup
            print(f'MASTER_ADDR: {os.environ['MASTER_ADDR']}')
            print(f'MASTER_PORT: {os.environ['MASTER_PORT']}')
            print(f'WORLD_SIZE: {os.environ['WORLD_SIZE']}')
            print(f'RANK: {os.environ['RANK']}')

            if env_vars:
                for k,v in env_vars.items():
                    os.environ[k] = str(v)

            # Only run the `torchrun` command on rank 0
            rank = int(os.environ['RANK'])
            if rank == 0:
                # Update the command to explicitly use the dynamic port
                full_command = (
                    f'torchrun --nproc_per_node={gpus_per_node} '
                    f'--rdzv_endpoint={master_addr}:{master_port} '
                    f'{command}'
                )

                print(f'Running command: {full_command}')
                subprocess.run(full_command, shell=True, check=True)
    else:
        def job_function():
            import os

            if env_vars:
                for k,v in env_vars.items():
                    os.environ[k] = str(v)

            print(f'Running command: {command}')
            subprocess.run(command, shell=True, check=True)


    # Submit the job
    job = executor.submit(job_function)

    print(f'Job submitted with ID: {job.job_id}, {job}')
    return job


def simulate_local_submitit_job(
    command: str,
    working_dir: str,
    log_dir: str,
    env_vars: Optional[dict[str, str]] = None,
) -> submitit.Job:
    """
    Simulate execution of a submitit.Job instance.

    Args:
      command: The command string to run.
      working_dir: The working directory for the subprocess.
      log_dir: Dir where submitit writes logs
    
    Returns:
      A mock Job object for the local CPU job.
    """
    # Create unique folder for the local job and get unique hash
    job_folder, job_id = fs_utils.create_unique_temp_folder(os.path.join(log_dir, 'local'), 'job')
    job = submitit.Job(folder=log_dir, job_id=job_id, tasks=(0,))
    
    # Override job._paths with the expected attributes.
    job._paths = SimpleNamespace(
        folder=job_folder,
        stdout=job_folder / "stdout",
        stderr=job_folder / "stderr",
        result_pickle=job_folder / "result.pkl",
        submitted_pickle=job_folder / "submitted.pkl"
    )
    
    job.watcher.get_state = lambda job_id, mode="standard": "RUNNING"

    # Run the command in the provided working directory.
    job_env = os.environ.copy()  # Start with a copy of the current environment
    if env_vars:
        job_env.update(env_vars)
    with job._paths.stdout.open("w") as out_f, job._paths.stderr.open("w") as err_f:
        process = subprocess.Popen(
            command,
            cwd=str(working_dir),
            stdout=out_f,
            stderr=err_f,
            shell=True,  # command must be provided as a list
            env=job_env
        )
        process.wait()
        retcode = process.returncode

    if retcode == 0:
        # Command succeeded: write the result pickle.
        outcome = "success"
        result = "Job completed successfully."
        with job._paths.result_pickle.open("wb") as f:
            pickle.dump((outcome, result), f)
        job.watcher.get_state = lambda job_id, mode="standard": "COMPLETED"
    else:
        # Command failed: ensure no result pickle exists.
        if job._paths.result_pickle.exists():
            job._paths.result_pickle.unlink()
        # Patch the job's watcher so that done() returns True (job is finished as "FAILED").
        job.watcher.get_state = lambda job_id, mode="standard": "FAILED"
    
    return job


async def main():
    def test_callback(job_result: JobResult):
        log_out = job_result.log_out[0]
        log_err = job_result.log_err[0]

        metrics = {}
        matches = re.findall(r'step:(\d+)(?:/\d+)?\s+val_loss:([\d.]+)\s+train_time:(\d+)ms', log_out)
        if matches:
            # Take the last match
            last_match = matches[-1]
            metrics = {
                'n_steps': int(last_match[0]),
                'val_loss': float(last_match[1]),
                'train_time': int(last_match[2])
            }

        print('Job completed')
        print('Results:')
        print(dataclasses.asdict(job_result))
        print('\nmetrics:')
        print(metrics)

    NANOGPT_ENV_VARS = {
        'NANOGPT_TRAIN_FILES': '/checkpoint/maui/minqijiang/data/fineweb10B/fineweb_train_*.bin',
        'NANOGPT_VAL_FILES': '/checkpoint/maui/minqijiang/data/fineweb10B/fineweb_val_*.bin',
        'NANOGPT_VAL_TOKENS': 10485760
    }

    # job = submit_job(
    #     command='test.py', 
    #     nodes=1, 
    #     gpus_per_node=1, 
    #     cpus_per_task=8, 
    #     tasks_per_node=1, 
    #     job_ttl=10,
    #     job_name='test_slurm',
    #     account='maui',
    #     working_dir='.'
    # )

    job = submit_job(
        command='train_gpt.py', 
        job_name='nanogpt',
        nodes=1,
        cpus_per_task=12,
        gpus_per_node=8,
        tasks_per_node=8,
        job_ttl=60,
        account='maui',
        working_dir='workspace/v_0',
        env_vars=NANOGPT_ENV_VARS,
    )

    JobObserver.shared.observe(
        job=job,
        metadata={'hypothesis': 'test hypothesis'},
        log_dir='submitit_logs',
        callback=test_callback,
    )

    await JobObserver.shared.wait()


if __name__ == '__main__':
    asyncio.run(main())
