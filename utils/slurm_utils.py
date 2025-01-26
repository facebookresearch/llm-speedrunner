from enum import Enum

import asyncio
import dataclasses
import re
import submitit
import subprocess


def launch_job(
    command: str,
    nodes: int,
    cpus_per_task: int,
    gpus_per_node: int,
    tasks_per_node: int,
    timeout_min: int,
    job_name: str,
    account: str,
    qos: str,
    working_dir: str,
    log_dir='submitit_logs'
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
    # Create a Submitit executor
    executor = submitit.AutoExecutor(folder=log_dir)

    # SLURM configuration
    executor.update_parameters(
        nodes=nodes,
        cpus_per_task=cpus_per_task,
        gpus_per_node=gpus_per_node,
        tasks_per_node=tasks_per_node,
        timeout_min=timeout_min,
        job_name=job_name,
        slurm_partition="gpu",  # Change this if a different partition is required
        slurm_account=account,
        slurm_qos=qos,
        slurm_additional_parameters={
            "chdir": working_dir
        },
    )

    # Define the job to run
    def job_function():
        import os
        os.system(f"{command}")

    # Submit the job
    job = executor.submit(job_function)

    print(f"Job submitted with ID: {job.job_id}")
    return job.job_id


class JobStatus(Enum):
    MISSING = 'missing'
    RUNNING = 'running'
    FAILURE = 'failure'
    SUCCESS = 'success'


@dataclasses.dataclass
class JobResult:
    status: JobStatus
    log_out: str
    log_err: str


class JobObserver:
    def __init__(self):
        """Manages a pool of asyncio tasks for observing submitit job status."""

    def observe(
        self,
        job: submitit.Job,
        log_dir: str,
        callback: Callable[JobResult],
    ):
        """Observe the status of a slurm job, and execute callback when finished.

        Args:
            job_id (str): SLURM job ID to check.
            callback (Callable[JobResult]): Called after job finishes.
        """
        pass

    async def wait(self):
        """Returns only when all pending jobs and their callbacks complete execution."""
        pass


def get_logs_out(job_id: str):
    pass


def get_logs_err(job_id: str):
    pass

