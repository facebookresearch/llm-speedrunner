# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from core.workspace import Workspace


def main():
	ws = Workspace(root_path='workspaces/nanogpt_10112024_20250209_144436_337805')

	info = ws.get_version_info('2')
	history_from_version = info.stable_ancestor_version

	relevant_history = ws.view_history(
		from_version=history_from_version,
		max_len=3,
		incl_good_versions=False,
		incl_buggy_versions=True,
		incl_ancestors=False,
		incl_descendents=True,
		descendent_depth=1,
		as_string=True
	)

	print(relevant_history)


if __name__ == '__main__':
	main()