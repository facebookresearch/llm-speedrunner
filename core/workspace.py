# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, Optional, Union, Type

import dataclasses
import logging
import json
import os
import re
import shutil
import subprocess
import time

import numpy as np

from core.types import ExperimentConfig
from utils import fs_utils


VERSION_REGEX = re.compile(r'^v_(\d+)$')


VERSION_SUMMARY_HEADER = """Version: {version}
Parent version: {parent_version}"""

VERSION_SUMMARY_TEMPLATE = """Hypothesis: {hypothesis}
Results: 
{metrics}
Has bugs? {has_bugs}
Outcome summary:
{outcome_summary}
"""


@dataclasses.dataclass
class VersionInfo:
    version: str  # e.g '1', '2', etc
    results: dict
    bug_depth: int = 0  # 0 if not buggy, 1 if buggy and parent is good, 2 if parent is also buggy, etc
    parent_version: Optional[str] = None
    children: Optional[list[str]] = None
    stable_ancestor_version: Optional[str] = None

    def get_summary_string(self, with_version_headers=True, with_private_metrics=False):
        metrics_dict = self.results.get('metrics', {})

        if not with_private_metrics and 'private' in metrics_dict:
            del metrics_dict['private']

        metrics_str = '\n'.join(
            [f'\t{k}: {v}' for k,v in metrics_dict.items()]
        )

        outcome_summary = self.results.get('outcome_summary', '')

        if with_version_headers:
            version_header = VERSION_SUMMARY_HEADER.format(
                version=self.version, 
                parent_version=self.parent_version,
            )
        else:
            version_header = ''

        summary = version_header + '\n' + VERSION_SUMMARY_TEMPLATE.format(
            hypothesis=self.results.get('hypothesis', ''),
            metrics=metrics_str,
            outcome_summary=outcome_summary,
            has_bugs='Yes' if self.bug_depth > 0 else 'No'
        )

        return summary


class Workspace:
    """Global workspace for the scientist. Tracks relevant artifacts.

    Workspace contains:
        - Directory reference to project files
        - Evaluation metrics per experiment
    """
    def __init__(
        self,
        root_path: str,
        template_dir: Optional[str] = None,
        packages: Optional[list[str]] = None,
        ignore_list: Optional[list[str]] = None,
    ):
        self.root_path: str = fs_utils.expand_path(root_path)
        os.makedirs(self.root_path, exist_ok=True)

        # Initialize version dirs
        self.template_dir = template_dir
        self.packages = packages if packages else []
        self.ignore_list = ignore_list

        version_dirs = self._get_version_dirs()
        self.n_versions = len(version_dirs)

        # Load version infos into memory
        self.version_infos = {info.version: info for info in self.load_version_info()}
        if self.version_infos:
            self.max_version = max([int(version) for version in self.version_infos])
        else:
            self.max_version = None

        if self.n_versions == 0:
            self.create_version(from_path=template_dir, ignore_list=[])

    def _get_version_dirs(self) -> list[str]:
        version_dirs = []

        for dirname in os.listdir(self.root_path):
            abs_dir_path = self.resolve_path(dirname)
            match = VERSION_REGEX.match(dirname)  # Match pattern and extract integer
            if os.path.isdir(abs_dir_path) and match:
                version_dirs.append((match.group(1), abs_dir_path))  # Store (integer, path)

        version_dirs.sort(key=lambda x: x[0])

        return {version: path for version, path in version_dirs}

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

    def load_version_info(self, version: Optional[str] = None) -> VersionInfo | list[VersionInfo]:
        """Return info for version, or all version infos if no version specified."""

        if version is None:
            versions = list(self._get_version_dirs().keys())
        else:
            versions = [version]

        infos = []
        for version in versions:
            try:
                results = json.loads(
                    self.view(
                        'results.json', 
                        version=version, 
                        no_filename_headers=True
                    ).strip()
                )
            except:
                results = {}

            try:
                meta = json.loads(
                        self.view('meta.json',
                        version=version,
                        no_filename_headers=True
                    ).strip()
                )
            except:
                meta = {}

            parent_version = meta.get('parent', None)
            try:
                parent_meta = json.loads(
                        self.view('meta.json',
                        version=parent_version,
                        no_filename_headers=True
                    ).strip()
                )
            except:
                parent_meta = {}
            parent_stable_ancestor_version = parent_meta.get('stable_ancestor_version', '0')

            infos.append(VersionInfo(
                    version=version,
                    results=results,
                    bug_depth=meta.get('bug_depth', 0),
                    parent_version=parent_version,
                    children=meta.get('children', None),
                    stable_ancestor_version=meta.get('stable_ancestor_version', parent_stable_ancestor_version)
                )
            )

        if isinstance(version, str) and len(infos) == 1:
            return infos[0]
        else:
            return infos

    def get_version_info(self, version: str) -> Optional[VersionInfo]:
        return self.version_infos.get(version)

    def save_to_file(self, text: str, path: str, version:Optional[str] = None):
        """Save text content to a file path in root_path."""
        save_path = self.resolve_path(path, version=version)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        with open(save_path, 'w') as fout:
            fout.write(text)

        if version is not None:  # Refresh version info if necessary
            self.version_infos[version] = self.load_version_info(version=version)

    def create_version(
        self, 
        from_path: Optional[str] = None, 
        from_version: Optional[str] = None,
        ignore_list: Optional[str] = None
    ) -> int:
        """Create new version directory, copying all contents in from_path."""
        # new_version = str(self.n_versions)
        self.n_versions += 1
        if self.max_version is None:
            self.max_version = 0
        else:
            self.max_version += 1
        new_version = str(self.max_version)
        new_version_dir_path = self.resolve_path(version=new_version)

        os.makedirs(new_version_dir_path, exist_ok=True)

        src_path = None
        if from_path is not None:
            src_path = fs_utils.expand_path(from_path)
        elif from_version is not None:
            src_path = self.resolve_path(version=from_version)
        elif self.template_dir is not None:
            src_path = fs_utils.expand_path(self.template_dir)

        if src_path is not None:
            if ignore_list is None:
                ignore_list = self.ignore_list
            fs_utils.cp_dir(src_path, new_version_dir_path, ignore_list=ignore_list)

        if from_version is not None:
            # Add a meta.json file with parent info to new version
            metadata = dict(parent=from_version, children=None, created_at=int(time.time()))
            self.save_to_file(json.dumps(metadata), path='meta.json', version=new_version)

            # Update parent's metadata with child pointer
            try:
                parent_metadata = json.loads(
                        self.view('meta.json',
                        version=from_version,
                        no_filename_headers=True
                    ).strip()
                )
            except:
                parent_metadata = {}

            children = parent_metadata.get('children')
            if children is None:
                children = [new_version]
            else:
                children.append(new_version)
            parent_metadata['children'] = children

            self.save_to_file(json.dumps(parent_metadata), path='meta.json', version=from_version)
            self.version_infos[from_version] = self.load_version_info(version=from_version)

        self.version_infos[new_version] = self.load_version_info(version=new_version)

        return new_version

    def delete_version(self, version: str):
        """Delete a version and all its children."""
        assert version != '0', 'Cannot delete root version.'

        children = []
        if version in self.version_infos:
            children = self.version_infos[version].children
            parent_version = self.version_infos[version].parent_version
            del self.version_infos[version]
            self.max_version = max([int(version) for version in self.version_infos])

        version_path = self.resolve_path(version=version)
        if os.path.exists(version_path):
            shutil.rmtree(version_path)
            self.n_versions -= 1

        # Update parent's child pointer
        parent_meta_path = self.resolve_path('meta.json', version=parent_version)
        if os.path.exists(parent_meta_path):
            with open(parent_meta_path, 'r') as f:
                metadata = json.load(f)
            metadata['children'] = [x for x in metadata['children'] if x != version]
            self.save_to_file(json.dumps(metadata), 'meta.json', version=parent_version)
            self.version_infos[parent_version] = self.load_version_info(version=parent_version)

        if children:
            for child_version in children:
                self.delete_version(child_version)

    def mark_as_buggy_from_version(self, version: str, from_version: Optional[str] = None):
        if from_version:
            meta = json.loads(
                self.view(
                    'meta.json',
                    version=version,
                    no_filename_headers=True
                ).strip()
            )
            parent_meta = json.loads(
                self.view(
                    'meta.json', 
                    version=from_version, 
                    no_filename_headers=True
                ).strip()
            )
        else:
            meta = {}
            parent_meta = {}
        meta['bug_depth'] = parent_meta.get('bug_depth', 0) + 1
        meta['stable_ancestor_version'] = parent_meta.get('stable_ancestor_version', from_version)
        self.save_to_file(json.dumps(meta), 'meta.json', version=version)

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
        recursive: bool = False,
        no_filename_headers: bool = False,
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
            if not no_filename_headers:
                contents.append(f'# {rel_path}')
            contents.append(file_content + '\n')

        return '\n'.join(contents)

    def exec_cmd(self, cmd: str):
        return subprocess.run(cmd, shell=True, capture_output=True, text=True)

    def get_top_k_versions(
        self, 
        selection_metric: str,
        from_versions: Optional[list[str]] = None,
        lower_is_better=False,
        k=1
    ) -> list[VersionInfo]:
        """Get the top-k versions based on a selection_fn
        
        Args:
            selection_fn: Given a results dict, returns a fitness score.
            from_versions: Choose only among these versions.
            k: Select the top-k versions.

        Returns:
            A list of the top-k versions based on the selection func.
        """
        if from_versions is None:
            version_infos = list(self.version_infos.values())
        else:
            from_versions_set = set(from_versions)
            version_infos = [
                info for version, info in self.version_infos.items() 
                if version in from_versions_set
            ]

        flip_coef = -1 if lower_is_better else 1
        default_value = -np.inf

        valid_infos = []
        scores = []
        for info in version_infos:
            metrics = info.results.get('metrics', {})

            score = metrics.get(selection_metric, None)

            if not metrics.get('is_valid', True):
                continue

            if score is None:
                score = default_value
            elif not isinstance(score, float):
                try:
                    score = float(score)
                except (ValueError, TypeError):
                    continue

            if score != default_value:
                score *= flip_coef

            scores.append(score)
            valid_infos.append(info)
        
        # Sort versions based on scores in desc order, with higher versions taking precedence if scores are equal
        sorted_infos = [
            info for info, _ in sorted(
                zip(valid_infos, scores), 
                key=lambda x: (x[1], int(x[0].version)), reverse=True
            )
        ]

        return sorted_infos[:k]

    def get_buggy_versions(self, is_leaf=True, max_bug_depth: Optional[int] = None) -> list[VersionInfo]:
        """Return all versions where is_buggy=True and <= max_bug_depth."""
        return [
            info for _, info in self.version_infos.items()
            if info.bug_depth > 0
            and (not is_leaf or info.children is None)
            and (max_bug_depth is None or info.bug_depth <= max_bug_depth)
        ]

    def get_good_versions(self) -> list[VersionInfo]:
        """Return all versions that are not buggy."""
        return [info for _, info in self.version_infos.items() if info.bug_depth == 0]

    def get_pending_versions(self) -> list[VersionInfo]:
        """Return all versions without results yet."""
        return [info for _, info in self.version_infos.items() if not info.results]

    def get_completed_versions(self) -> list[VersionInfo]:
        """Return all versions with results."""
        return [info for _, info in self.version_infos.items() if info.results and info.version != '0']

    def view_history(
        self,
        from_version: Optional[str] = None, 
        max_len: Optional[int] = None, 
        incl_good_versions=True,
        incl_buggy_versions=False,
        incl_ancestors=True,
        incl_descendents=False,
        ancestor_depth: Optional[int] = None,
        descendent_depth: Optional[int] = None,
        as_string=True
    ) -> str | list[str]:
        """Return recent VersionInfo instances summarized as a single string.

        Results are returned with most recent versions first.

        Args:
            from_version: If None, returns all version infos subject to filter conditions.
                Note in this case, ancestor and descendent-related conditions are ignored.
            max_len: Maximum number of recent versions to consider.
            incl_good_versions: Whether to include good versions.
            incl_buggy_versions: Whether to include buggy versions.

            incl_ancestors: Whether to return nodes from which from_version descends.
            incl_descendents: Whether to return nodes descending from from_version.
            ancestor_depth: If not None, only return ancestors up to this many degrees away.
            descendent_depth: If not None, only return descendents up to this many degrees away.
            as_string: Whether to return the history as a string.

        Returns:
            A string summary or list of most recent, filtered version infos.
        """
        if from_version is None:
            infos = list(self.version_infos.values())
        else:
            if not from_version in self.version_infos:
                raise ValueError(f'Version {from_version} is missing.')

            from_version_info = self.version_infos[from_version]

            # Get ancestors if necessary
            infos = []
            parent_version = from_version_info.parent_version
            depth = 0
            while parent_version is not None and incl_ancestors and (
                not ancestor_depth or depth < ancestor_depth
            ):
                parent = self.version_infos.get(parent_version)
                infos.append(parent)
                depth += 1
                parent_version = parent.parent_version

            # Get descendents via BFS
            descend_versions = from_version_info.children
            while descend_versions and incl_descendents and (
                not descendent_depth or depth < descendent_depth
            ):
                descend_infos = [self.version_infos.get(v) for v in descend_versions]
                infos += descend_infos

                descend_versions = []
                for info in descend_infos:
                    if info.children:
                        descend_versions += info.children

                depth += 1

        # Filter away good/buggy versions
        infos = [
            x for x in infos
            if (incl_good_versions and not x.bug_depth > 0) 
            or (incl_buggy_versions and x.bug_depth > 0)
        ]

        sorted_infos = sorted(infos, key=lambda x: x.version, reverse=True)

        if max_len:
            sorted_infos = sorted_infos[:max_len]

        if as_string:
            summary = '\n'.join(
                [f'<info>{x.get_summary_string()}</info>' for x in sorted_infos]
            )

            if len(sorted_infos) > 0:
                header = '<version_log>'
                footer = '</version_log>'
                summary = f'{header}\n{summary}\n{footer}'

            return summary
        else:
            return sorted_infos
