# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Compute code embedding distances between LLM solutions and 
current and next ground-truth human records.

Usage:

Compute embeddings:
CUDA_VISIBLE_DEVICES=0 python plot/analyze_embeddings.py \
--json_path=/home/user123456/scientist/code_analysis_with_all_versions_knowledge.json \
--save_path=/path/to/saved/df.csv

Plot distances:
python plot/analyze_embeddings.py \
--df_path=/home/user123456/nanogpt_embed_df.csv \
--plot_metric='l2_end_mean' \
--models 'o3-mini' \
--levels '12' '125'\
--ylabel 'L2 distance'

python plot/analyze_embeddings.py \
--df_path=/home/user123456/nanogpt_embed_df.csv \
--plot_metric='cos_end_mean' \
--models 'o3-mini' \
--levels '12' '125'\
--ylabel 'Cosine distance'
"""

import argparse
import json
import os
from typing import Dict, List, Any

import pandas as pd
import torch
import torch.nn.functional as F
# from transformers import AutoModel
import tqdm
import matplotlib.pyplot as plt


THRESHOLD_VAL_LOSS = 3.28
AutoModel = Any

def load_json(path: str) -> Dict:
    with open(path, "r") as fp:
        return json.load(fp)


def get_embedding(
    model: AutoModel,
    text: str,
    cache: Dict[str, torch.Tensor],
    max_length: int = 32768,
) -> torch.Tensor:
    if text not in cache:
        emb = model.encode_corpus([text], max_length=max_length)[0].detach()
        cache[text] = emb  # save raw vector
    return cache[text]


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return (
        F.cosine_similarity(F.normalize(a, dim=0), F.normalize(b, dim=0), dim=0)
        .item()
    )


def l2(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.norm(a - b, p=2).item()


def _count_runs(data: Dict) -> int:
    """Return number of run records in the whole JSON."""
    return sum(
        len(records_dict)
        for levels_dict in data.values()
        for records_dict in levels_dict.values()
    )


def build_dataframe(data: Dict, model: AutoModel) -> pd.DataFrame:
    rows: List[Dict] = []
    cache: Dict[str, torch.Tensor] = {}

    total_runs = _count_runs(data)
    with tqdm.tqdm(total=total_runs, desc="Computing embedding distances", unit="run") as pbar:
        for method_name, levels_dict in data.items():  # tree / forest / flat / …
            for levels_key, records_dict in levels_dict.items():  # "12", "125", …
                for run_id, record in records_dict.items():  # record_{n}-<hash>
                    # group-key information
                    record_idx = record.get("record")
                    model_name = record.get("model", "")
                    levels_value = levels_key

                    # locate the “solution” version v_i whose metrics == record[metrics]
                    llm_code = None

                    best_code = None
                    best_time = float("inf")
                    fallback_code = None
                    fallback_time = float("inf")

                    for k, v in record.items():
                        if not k.startswith("v_"):
                            continue

                        m = v.get('metrics')
                        val_loss = m.get("val_loss")
                        if val_loss is None:
                            val_loss = float("inf")
                        runtime  = m.get("train_time")
                        if runtime is None:
                            runtime = float("inf")

                        if val_loss <= THRESHOLD_VAL_LOSS and runtime < best_time:
                            best_code, best_time = v["code"], runtime

                        if runtime < fallback_time:
                            fallback_code, fallback_time = v["code"], runtime

                    llm_code = best_code if best_code is not None else fallback_code

                    if llm_code is None:  # no exact metrics match
                        print(f'{run_id}: No well-formed run records found, skipping!')
                        continue

                    # fetch source / target code strings
                    human_code = record["human_code"]
                    next_human_code = record["next_human_code"]

                    # embeddings (cached)
                    e_llm = get_embedding(model, llm_code, cache)
                    e_human = get_embedding(model, human_code, cache)
                    e_next = get_embedding(model, next_human_code, cache)

                    # distances
                    cos_start = 1 - cosine(e_llm, e_human)
                    cos_end = 1 - cosine(e_llm, e_next)
                    l2_start = l2(e_llm, e_human)
                    l2_end = l2(e_llm, e_next)

                    rows.append(
                        dict(
                            method=method_name,
                            model=model_name,
                            record=record_idx,
                            levels=levels_value,
                            run_id=run_id,
                            cos_start=cos_start,
                            cos_end=cos_end,
                            l2_start=l2_start,
                            l2_end=l2_end,
                        )
                    )

                    pbar.update(1)

    return pd.DataFrame(rows)


def aggregate_and_print(df: pd.DataFrame) -> None:
    grouped = (
        df.groupby(["method", "model", "record", "levels"])
        .agg(
            cos_start_mean=("cos_start", "mean"),
            cos_end_mean=("cos_end", "mean"),
            l2_start_mean=("l2_start", "mean"),
            l2_end_mean=("l2_end", "mean"),
        )
        .reset_index()
    )
    pd.set_option("display.max_rows", None)
    print(grouped.to_string(index=False))

    return grouped


def _resolve(path: str | None) -> str | None:
    """Expand '~' and convert to absolute path (leave None unchanged)."""
    if path is None:
        return None
    return os.path.abspath(os.path.expanduser(path))


def main():
    parser = argparse.ArgumentParser(
        description="Compute embedding-space distances between LLM and ground-truth human code blocks."
    )
    parser.add_argument("--json_path", type=str, help="Path to big JSON file.")
    parser.add_argument("--save_path", type=str, help="Where to write the raw-distance dataframe (.csv).")
    parser.add_argument("--df_path", type=str, help="If supplied, load this DF instead of recomputing from JSON.")
    parser.add_argument(
        "--plot_metric",
        type=str,
        default=None,
        help="Name of the aggregated metric to plot (e.g. cos_start_mean)."
    )
    parser.add_argument(
        "--levels",
        nargs="+",
        type=str,
        default=None,
        help="Which hint-levels to include in the plot (e.g. 12 125)."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        type=str,
        default=None,
        help="Which models to include."
    )
    parser.add_argument(
        "--ylabel",
        type=str,
        default='Distance',
        help="y-axis label."
    )
    args = parser.parse_args()

    # Resolve paths
    json_path = _resolve(args.json_path)
    save_path = _resolve(args.save_path)
    df_path   = _resolve(args.df_path)

    # Load or build the df
    if df_path and os.path.exists(df_path):
        print(f"Loading existing dataframe from {df_path}")
        df = pd.read_csv(df_path)
    else:
        if not json_path or not os.path.exists(json_path):
            raise FileNotFoundError(
                "--json_path must be supplied (and exist) when --df_path is not provided."
            )
        model = AutoModel.from_pretrained(
            "Salesforce/SFR-Embedding-Code-2B_R", trust_remote_code=True
        )
        data = load_json(json_path)
        df = build_dataframe(data, model)

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            print(f"Saving raw distances to {save_path}")
            df.to_csv(save_path, index=False)

    print("\nAggregated means:")
    df = aggregate_and_print(df)

    if args.plot_metric:
        levels = args.levels or ["12", "125"]
        try:
            levels = [int(l) for l in levels]
        except ValueError:
            raise ValueError(f"Could not parse levels {levels} as integers")

        models = args.models
        plot_df = df[df["levels"].isin(levels)]
        if models is not None: 
            plot_df = plot_df[plot_df["model"].isin(models)]

        if plot_df.empty:
            raise ValueError("Nothing to plot after filtering by levels/models")

        # Markers keyed by (model, levels)
        marker_cycle = ["o", "s", "D", "x", "^", "v", "<", ">", "P", "X"]
        uniq_pairs = plot_df[["model", "levels"]].drop_duplicates()
        pair_markers = {
            (row.model, row.levels): marker_cycle[i % len(marker_cycle)]
            for i, row in uniq_pairs.reset_index(drop=True).iterrows()
        }

        cmap = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        methods = plot_df["method"].unique()
        method_colors = {m: cmap[i % len(cmap)] for i, m in enumerate(methods)}

        plt.figure()
        for (meth, mdl, lvl), sub in plot_df.groupby(["method", "model", "levels"]):
            sub = sub.sort_values("record")
            if sub.empty:
                continue
            plt.plot(
                sub["record"],
                sub[args.plot_metric],
                marker=pair_markers[(mdl, lvl)], 
                color=method_colors[meth],
                linestyle="-",
                label=f"{meth}, {mdl}, {lvl}"
            )

        max_record = int(plot_df["record"].max())
        plt.xticks(range(1, max_record + 1))

        plt.xlabel("Record index")
        plt.ylabel(args.ylabel)
        plt.legend()
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
