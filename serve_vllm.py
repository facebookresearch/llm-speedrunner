# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import submitit


def run_vllm_server():
    import subprocess
    model_path = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
    command = [
        "vllm",
        "serve",
        model_path,
        "--gpu-memory-utilization", "0.9",
        "--tensor-parallel-size", "2",
        "--enable-prefix-caching"
    ]
    subprocess.run(command, check=True)


def main():
    executor = submitit.AutoExecutor(folder="submitit_logs/vllm_server")
    executor.update_parameters(
        timeout_min=60*12,
        gpus_per_node=2,
        cpus_per_task=4,
        mem_gb=70,
        slurm_account="ram",
        slurm_qos="dev"
    )

    # Submit the job
    job = executor.submit(run_vllm_server)
    print(f"Job submitted with ID: {job.job_id}")


if __name__ == "__main__":
    main()
