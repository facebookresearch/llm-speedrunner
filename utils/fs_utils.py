# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Tuple
from pathlib import Path
import fnmatch
import os
import tempfile
import shutil


def expand_path(path: str) -> str:
    """Expands path into an absolute path."""
    return os.path.abspath(os.path.expanduser(path))


def fname_matches_any(fname: str, patterns: Optional[list[str]] = None) -> bool:
    if not patterns:
        return False

    for pattern in patterns:
        if fnmatch.fnmatch(fname, pattern):
            return True

    return False


def cp_dir(src_dir: str, target_dir: str, ignore_list: Optional[list[str]] = None,):
    """
    Copies all files and directories from the source directory to the target directory,
    preserving the directory structure.

    Args:
        src_dir (str): Path to the source directory.
        target_dir (str): Path to the target directory.
        ignore_list: (list[str]): A list of base dirnames and filenames to ignore.

    Raises:
        ValueError: If src_dir does not exist or is not a directory.
    """
    src_dir = os.path.abspath(os.path.expanduser(src_dir))
    target_dir = os.path.abspath(os.path.expanduser(target_dir))
    if ignore_list is None:
        ignore_list = []

    if not os.path.isdir(src_dir):
        raise ValueError(f"Source directory '{src_dir}' does not exist or is not a directory.")

    # Walk through the source directory
    for root, dirs, files in os.walk(src_dir):
        relative_path = os.path.relpath(root, src_dir)
        target_path = os.path.join(target_dir, relative_path)
        os.makedirs(target_path, exist_ok=True)

        # Copy all files in the current directory
        for file in files:
            if fname_matches_any(os.path.basename(file), ignore_list):
                continue

            src_file = os.path.join(root, file)
            dest_file = os.path.join(target_path, file)
            shutil.copy2(src_file, dest_file)

        # Ensure dirs are created in the target
        for dir_name in dirs:
            if fname_matches_any(os.path.basename(dir_name), ignore_list):
                continue

            src_subdir = os.path.join(root, dir_name)
            target_subdir = os.path.join(target_path, dir_name)
            os.makedirs(target_subdir, exist_ok=True)


def create_unique_temp_folder(parent_dir: str, name: str) -> Tuple[Path, str]:
    """
    Create a unique temporary folder under <submitit_log_dir>/local/ using the given name as a prefix.
    Returns both the full folder path and the unique hash part (the suffix after the prefix).

    Args:
      submitit_log_dir (str or Path): The base directory.
      name (str): The prefix for the folder name.

    Returns:
      tuple(Path, str): (full folder Path, unique hash as a string)
    """
    base_dir = Path(parent_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    
    # Create the directory. mkdtemp returns a full path that starts with name + '_'
    full_folder = tempfile.mkdtemp(prefix=name + "_", dir=str(base_dir))
    full_folder_path = Path(full_folder)
    
    prefix = name + "_"
    unique_hash = full_folder_path.name[len(prefix):]
    
    return full_folder_path, unique_hash
