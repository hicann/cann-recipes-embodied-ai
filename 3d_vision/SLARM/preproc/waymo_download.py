# coding=utf-8
# Adapted from
# https://github.com/NVlabs/GaussianSTORM/blob/main/preproc/waymo_download.py

# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) NVIDIA. All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from typing import List


def download_file(filename, target_dir, source):
    result = subprocess.run(
        [
            "gsutil",
            "cp",
            "-n",
            f"{source}/{filename}.tfrecord",
            target_dir,
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise Exception(result.stderr)


def download_files(
    file_names: List[str],
    target_dir: str,
    source: str,
    max_workers: int = 10,
):
    total_files = len(file_names)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(download_file, filename, target_dir, source) for filename in file_names
        ]

        for counter, future in enumerate(futures, start=1):
            try:
                future.result()
                print(f"[{counter}/{total_files}] Downloaded successfully!")
            except Exception as e:
                print(f"[{counter}/{total_files}] Failed to download. Error: {e}")


if __name__ == "__main__":
    print("note: `gcloud auth login` is required before running this script")
    print("Downloading Waymo dataset from Google Cloud Storage...")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target_dir",
        type=str,
        default="data/waymo/raw",
        help="Path to the target directory",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
    )
    parser.add_argument(
        "--scene_ids", type=int, nargs="+", help="scene ids to download", default=None
    )
    parser.add_argument(
        "--split_file",
        type=str,
        default="data/dataset_scene_list/waymo_train_list.txt",
        help="",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=10,
        help="Number of threads to use for downloading",
    )
    parser.add_argument(
        "--version",
        type=str,
        default="waymo_open_dataset_scene_flow",
        help="",
    )
    args = parser.parse_args()
    os.makedirs(args.target_dir, exist_ok=True)
    total_list = open(args.split_file, "r").readlines()
    total_list = [x.strip() for x in total_list]
    if args.scene_ids is not None:
        file_names = [total_list[i] for i in args.scene_ids]
    else:
        file_names = total_list
    download_files(
        file_names,
        args.target_dir,
        source=f"gs://{args.version}/{args.split}",
        max_workers=args.max_workers,
    )
