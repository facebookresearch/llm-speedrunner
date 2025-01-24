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
    )

    # Define the job to run
    def job_function():
        import os
        os.system(f"{command}")

    # Submit the job
    job = executor.submit(job_function)

    print(f"Job submitted with ID: {job.job_id}")
    return job.job_id


def check_job_status(job_id: str, log_folder: str) -> str | None:
    """
    Checks if a given SLURM job is completed. If completed, retrieves the .out file contents.
    
    Args:
        job_id (str): The SLURM job ID to check.
        log_folder (str): The path to the logs folder where the .out file is stored.

    Returns:
        str | None: The contents of the .out file if the job is completed, otherwise None.
    """
    try:
        # Use `sacct` to query the job status
        result = subprocess.run(
            ["sacct", "-j", job_id, "--format=JobID,State", "--noheader"],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            raise RuntimeError(f"sacct command failed: {result.stderr}")

        # Parse job state from the command output
        output = result.stdout.strip()
        if not output:
            print(f"Job {job_id} not found. It may still be in the queue.")
            return None

        job_status = output.split()[1]  # Second column is the state (e.g., COMPLETED, FAILED)
        if job_status in {"COMPLETED", "FAILED", "CANCELLED"}:
            # Job is finished; retrieve the .out file
            log_file = os.path.join(log_folder, f"{job_id}.out")
            if log_file.exists():
                return log_file.read_text()
            else:
                print(f"Log file {log_file} not found.")
                return None

        print(f"Job {job_id} is not completed yet. Current state: {job_status}")
        return None

    except Exception as e:
        print(f"Error checking SLURM job status: {e}")
        return None


class JobObserver:
	pass


