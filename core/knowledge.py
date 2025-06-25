# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional
import dataclasses
import os
import glob

from core.types import Serializable
from utils import fs_utils


@dataclasses.dataclass
class KnowledgeEntry:
    content: str
    metadata: Optional[dict[str, Serializable]] = None


class KnowledgeStore:
    def __init__(
        self,
        entries: Optional[list[str] | list[KnowledgeEntry]] = None,
        src_paths: Optional[list[str]] = None
    ):
        """Allows interfacing with knowledge sources via a common interface.

        Args:
            src_paths: A list of file paths or glob regex to load into the knowledge store.
            contents: A list of strings to add directly as entries into the knowledge store.
        """
        self._entries = []

        if entries:
            for entry in entries:
                self.insert(entry)

        if src_paths:
            for path in src_paths:
                abs_path = fs_utils.expand_path(path)

                if '*' in abs_path or '?' in abs_path:
                    match_files = glob.glob(abs_path)
                else:
                    match_files = [abs_path]

                for path in match_files:
                    if os.path.isfile(path):  # Ensure it's a valid file
                        with open(path, 'r') as f:
                            self.insert(f.read().strip())

    def insert(self, entry: str | KnowledgeEntry):
        """Insert an entry. (msj: Should eventually support deduping.)"""
        if isinstance(entry, str):
            entry = KnowledgeEntry(entry)
        self._entries.append(entry)

    def search(
        self,
        query: Optional[str] = None,
        max_len: Optional[int] = None,
        as_string=True
    ) -> list[KnowledgeEntry] | str:
        """Read from the knowledge store. 

        Args:
            query: Used to filter results in the store. 
            as_string: Whether to return all entries as a single formatted string.

        Returns:
            For simplicity, just return all entries for now, either as a list
            of KnowledgeEntry instances or a formatted string.
        """
        entries = self._entries
        if max_len is not None:
            entries = self._entries[:max_len]

        if as_string:
            summary = '\n'.join([f'<li>{x}</li>' for x in entries])

            if summary:
                head = '<knowledge>'
                footer = '</knowledge>'
                summary = f'{head}\n{summary}\n{footer}'

            return summary
        else:
            return entries
