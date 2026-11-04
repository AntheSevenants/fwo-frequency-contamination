import argparse
import json
import pandas as pd
import numpy as np

import export.sweeps
import export.runs
import export.combinations

import batch.combination
import batch.aggregate
import batch.posthoc

from tqdm.auto import tqdm
from collections import defaultdict

SWEEPS_DIR = "sweeps/"


def do_report(current_sweep: str):
    run_infos = export.sweeps.get_run_infos(SWEEPS_DIR, current_sweep)
    # Aggregate so only unique combinations remain
    report_df = run_infos.groupby("combination_id").first().reset_index()

    new_cols = defaultdict(list)
    for index, row in tqdm(
        report_df.iterrows(), total=len(report_df), desc="Computing report"
    ):
        combination_id = row["combination_id"]
        combination_data = export.combinations.get_combination_data(
            SWEEPS_DIR, current_sweep, combination_id
        )

        for result_column, operation in batch.posthoc.posthoc_operations.items():
            new_cols[result_column].append(operation(row, combination_data))

    for column in new_cols:
        report_df[column] = new_cols[column]

    # Write report to file
    report_path = export.sweeps.make_report_path(SWEEPS_DIR, current_sweep)
    report_df.to_csv(report_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="batch_run_report - build a quick report to describe model behaviour"
    )
    parser.add_argument("current_sweep", type=str, help="sweep ID")
    args = parser.parse_args()

    do_report(args.current_sweep)
