from typing import Optional
import os
import submitit
import subprocess


def launch_job(
    nodes: int,
    cpus_per_task: int,
    gpus_per_node: int,
    tasks_per_node: int,
    gpu_mem: int,
    timeout_min: int,
    job_name: str,
    account: str,
    qos: Optional[str] = None,
    working_dir: str = '.',
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
        slurm_account=account,
        slurm_qos=qos,
        slurm_additional_parameters={
            "chdir": working_dir,
            "gres": f"gpu:{gpus_per_node}",  # Request GPUs
        },
    )

    def job_function():
        import socket

        def find_free_port():
            """Find an available port on the system."""
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', 0))  # Bind to any available port
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                return s.getsockname()[1]  # Return the port number

        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

        # Set up SLURM environment variables
        master_addr = subprocess.getoutput("scontrol show hostname $SLURM_NODELIST | head -n 1").strip()
        os.environ["MASTER_ADDR"] = master_addr

        # Dynamically set the master port
        master_port = find_free_port()
        os.environ["MASTER_PORT"] = str(master_port)

        os.environ["WORLD_SIZE"] = str(int(nodes * tasks_per_node))  # Total processes
        os.environ["RANK"] = os.environ.get("SLURM_PROCID", "0")  # Rank assigned by SLURM

        # Debugging logs for distributed setup
        print(f"MASTER_ADDR: {os.environ['MASTER_ADDR']}")
        print(f"MASTER_PORT: {os.environ['MASTER_PORT']}")
        print(f"WORLD_SIZE: {os.environ['WORLD_SIZE']}")
        print(f"RANK: {os.environ['RANK']}")

        # Only run the `torchrun` command on rank 0
        rank = int(os.environ["RANK"])
        if rank == 0:
            # Update the command to explicitly use the dynamic port
            command = (
                f"torchrun --nproc_per_node={gpus_per_node} "
                f"--rdzv_endpoint={master_addr}:{master_port} "
                f"train_gpt.py"
            )

            print(f"Running command: {command}")
            subprocess.run(command, shell=True, check=True)

    # Submit the job
    job = executor.submit(job_function)

    print(f"Job submitted with ID: {job.job_id}")
    return job.job_id


if __name__ == '__main__':
    command = (
        "torchrun "
        "--nproc_per_node=8 "
        "train_gpt.py"
    )
    launch_job(
        job_name='nanogpt',
        nodes=1,
        cpus_per_task=12,
        gpus_per_node=8,
        tasks_per_node=8,
        timeout_min=60,
        account='maui',
        # qos='maui_high',
    )
