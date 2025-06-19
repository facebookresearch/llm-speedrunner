#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import glob
import json
import os
import subprocess


def main():
    data_dir = os.environ.get('GRADER_DATA_PATH')
    cmd = [
        "mlebench", "grade",
        "--submission", "submission.jsonl",
        "--output-dir", ".",
        "--data-dir", data_dir
    ]

    subprocess.run(
        cmd, 
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True,
    )

    reports = glob.glob("*grading_report.json")
    if reports:
        latest_report = sorted(reports)[-1]
        with open(latest_report, "r") as f:
            data = json.load(f)
            res_str = ', '.join([
                f'{k}: {v}' for k, v in data.items() 
                if k != 'competition_reports'
            ])
            res_str = f'score: {data['competition_reports'][0]['score']}, ' + res_str

    print(res_str, flush=True)


if __name__ == '__main__':
    main()
