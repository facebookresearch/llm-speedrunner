from typing import Optional, Union, Type

import datetime
import dataclasses
import logging
import os
import re
import shutil
import subprocess
import time

from core.types import ExperimentRecord, ExperimentHistory, ExperimentConfig
from utils import fs_utils


class Workspace:
    """Global workspace for the scientist. Tracks relevant artifacts.

    Workspace contains:
        - Directory reference to project files
        - Chain of experiment diffs from base project
        - Evaluation metrics per experiment
    """
    def __init__(self, root_path: str, template_dir: Optional[str] = None, track_history=True):
        self.root_path: str = fs_utils.expand_path(root_path)
        os.makedirs(self.root_path, exist_ok=True)

        if track_history:
            self._exp_history: ExperimentHistory = ExperimentHistory(records=[])
        else:
            self._exp_history = None

        # Initialize version dirs
        self.template_dir = template_dir

        version_dirs = self._get_version_dirs()
        self.n_versions = len(version_dirs)
        self.create_version(from_path=template_dir)

        # Copy files from cp_dir
        if template_dir is not None:
            fs_utils.cp_dir(template_dir, 
                self.resolve_path(version=str(self.n_versions))
            )

    def _get_version_dirs(self) -> list[str]:
        version_dirs = []
        pattern = re.compile(r'^v_(\d+)$')

        for dirname in os.listdir(self.root_path):
            abs_dir_path = self.resolve_path(dirname)
            match = pattern.match(dirname)  # Match pattern and extract integer
            if os.path.isdir(abs_dir_path) and match:
                version_dirs.append((int(match.group(1)), abs_dir_path))  # Store (integer, path)

        version_dirs.sort(key=lambda x: x[0])

        return [path for _, path in version_dirs]

    def _get_version_dirname(self, version: str) -> str:
        return f'v_{version}'

    def resolve_path(self, path='', version:Optional[str] = None) -> str:
        """Resolve relative path based on version, or root dir if no version."""
        if version is None:
            return os.path.join(self.root_path, path)
        else:
            return os.path.join(
                self.root_path,
                (os.path.join(self._get_version_dirname(version), path))
            )

    def save_to_file(self, text: str, path: str, version:Optional[str] = None):
        """Save text content to a file path in root_path."""
        save_path = self.resolve_path(path, version=version)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        with open(save_path, 'w') as fout:
            fout.write(text)

    def create_version(
        self, 
        from_path: Optional[str] = None, 
        from_version: Optional[str] = None
    ):
        """Create new version directory, copying all contents in from_path."""
        self.n_versions += 1
        new_version_dir_path = self.resolve_path(version=(str(self.n_versions)))

        os.makedirs(new_version_dir_path, exist_ok=True)
        if from_path is not None:
            src_path = fs_utils.expand_path(from_path)
        elif from_version is not None:
            src_path = self.resolve_path(version=from_version)
        elif self.template_dir is not None:
            src_path = fs_utils.expand_path(self.template_dir)

        if src_path is not None:
            fs_utils.cp_dir(src_path, new_version_dir_path)

        return self.n_versions

    def ls(self, path = ''):
        """List all files and directories at a path"""
        abs_path = self.resolve_path(path)
        if os.path.isdir(abs_path):
            return os.listdir(abs_path)
        else:
            raise ValueError(f'{path} is not a directory.')

    def view(
        self, 
        paths: Optional[Union[list[str], str]] = None, 
        version: Optional[str] = None, 
        recursive: bool = False
    ) -> str:
        """
        View the contents of the specified paths, concatenating file contents
        in the format "# relative_path\n<file_content>\n".

        If recursive=True, all files under the provided paths (including subdirectories)
        are concatenated in the order of the filesystem walk.

        Args:
            paths (list[str] | str | None): The paths to view. Defaults to None (root).
            version (str | None): The version directory to resolve paths against. Defaults to None.
            recursive (bool): If True, includes files from subdirectories. Defaults to False.

        Returns:
            str: Concatenated contents of all specified files.
        """
        if paths is None:
            paths = ['']
        elif isinstance(paths, str):
            paths = [paths]

        abs_paths = [self.resolve_path(path, version=version) for path in paths]
        all_files = {}

        for abs_path in abs_paths:
            if not os.path.exists(abs_path):
                raise ValueError(f'Path {abs_path} does not exist')

        for path in abs_paths:
            if os.path.isdir(path):
                if recursive:
                    for root, _, files in os.walk(path):
                        for file in files:
                            file_path = os.path.join(root, file)
                            all_files[file_path] = None
                else:
                    for maybe_file in os.listdir(path):
                        file_path = os.path.join(path, maybe_file)
                        if os.path.isfile(file_path):
                            all_files[file_path] = None
            elif os.path.isfile(path):
                all_files[path] = None

        for file_path in all_files:
            try:
                with open(file_path, 'r') as fin:
                    all_files[file_path] = fin.read()
            except:
                logging.error(f'Failed to open file {file_path}')
                continue

        base_dir = self.resolve_path(version=version)

        contents = []
        for abs_path, file_content in all_files.items():
            rel_path = os.path.relpath(abs_path, base_dir)
            contents.append(f'# {rel_path}')
            contents.append(file_content + '\n')

        return '\n'.join(contents)

    def exec_cmd(self, cmd: str):
        return subprocess.run(cmd, shell=True, capture_output=True, text=True)

    def view_history(
        self, 
        max_len: Optional[int] = None, 
        as_diffs=True, 
        valid_only=True
    ) -> str:
        """Return most recent experiment records summarized as a single string.

        Args:
            max_len: Only return this many of most recent records.
            as_diffs: If True, return as a chain of diffs from first returned record.

        Returns:
            A string summary of the max_len most recent experiment records.
        """
        # if track_history:
        #   return self._exp_history[-max_len:]
        # else:
        #   return None
        pass

    def save_to_history(self, record: ExperimentRecord):
        """Save a new experiment record to experiment history."""
        if not track_history:
            return

        self._exp_history.append(record)



