# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A dummy ideator agent that passes through knowledge without any model interactions.
"""
from typing import Optional

from core.agent import Agent
from core.workspace import Workspace


class DummyIdeator(Agent):
    def ideate(
        self,
        task_description: str,
        fnames: list[str],
        workspace: Workspace,
        version: int,
        ignore_ideas: Optional[str] = None,
        history: Optional[str] = None,
        knowledge: Optional[str] = None,
        max_retries=1
    ) -> tuple[list[str], Optional[dict[str, str]]]:
        """Pass through the knowledge without any modifications.
        
        Args:
            task_description: Description of the task (not used)
            fnames: List of filenames (not used)
            workspace: Workspace object (not used)
            version: Version number (not used)
            ignore_ideas: Ideas to ignore (not used)
            history: History string (not used)
            knowledge: Knowledge string to pass through
            max_retries: Maximum number of retries (not used)
            
        Returns:
            Tuple of (list of knowledge strings, metadata dict)
        """
        # If no knowledge provided, return empty list
        if not knowledge:
            return [], {"ideator_type": "dummy"}
            
        # Split knowledge into lines and return
        knowledge_lines = [line.strip() for line in knowledge.split('\n') if line.strip()]
        return knowledge_lines, {
            "summary": "Dummy ideator passed through knowledge",
            "ideator_type": "dummy",
            "num_knowledge_items": len(knowledge_lines)
        } 