# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from unittest.mock import patch, MagicMock
import json
import os
import pytest
import shutil

from scientist.core.types import ExperimentConfig
from scientist.core.workspace import Workspace


@pytest.fixture(scope="class")
def workspace_factory(tmp_path_factory):
    """
    Returns a factory function to create a Workspace with custom content.
    """
    tmp_dir = tmp_path_factory.mktemp("scientist_tests")

    def _create_workspace(template_contents: dict[str, str] = None):
        """Create a new Workspace in a temporary directory.

        Args:
            cp_dir_contents (Dict[str, str]): 
                A dictionary mapping relative file paths to file contents.
                Example:
                    {
                        "test.txt": "Some text",
                        "data/test_data.csv": "col1,col2\nval1,val2\n"
                    }

        Returns:
            (ws, root_path, template_dir): The Workspace instance and its paths.
        """
        # Create a unique temp directory for each call
        root_path = tmp_dir / "workspace"
        template_dir = tmp_dir / "template_dir"

        # Create template_dir and populate it if cp_dir_contents is provided
        if template_contents:
            for relative_path, file_contents in template_contents.items():
                dest_path = template_dir / relative_path
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                dest_path.write_text(file_contents)
        else:
            template_dir.mkdirs(parents=True, exist_ok=True)
            (template_dir / "default.txt").write_text("Default content")

        # Create the Workspace
        ws = Workspace(
            root_path=str(root_path),
            template_dir=str(template_dir),
        )

        return ws, root_path, template_dir

    yield _create_workspace

    shutil.rmtree(tmp_dir, ignore_errors=True)


def check_files_with_contents_exist(file_contents: dict[str, str]):
    for abs_path, expected_contents in file_contents.items():
        assert os.path.exists(abs_path), (
            f"File from {abs_path} should be copied to root_dir."
        )
        with open(abs_path, "r") as f:
            assert f.read() == expected_contents, (
                f"Copied file at {abs_path} content should match original."
            )


class TestNewWorkspace:
    """
    A test class that shares a single Workspace instance across all methods.
    """
    @pytest.fixture(autouse=True, scope="class")
    def _setup_class(self, workspace_factory):
        template_contents = {
            "test.txt": "These are the droids you are looking for.",
            "data/test_data.csv": "name,role\nnikola tesla,scientist\n"
        }
        type(self).template_contents = template_contents

        (
            type(self).workspace, 
            type(self).root_path, 
            type(self).template_dir
        ) = workspace_factory(template_contents)

    def test_root_directory_created(self):
        assert os.path.exists(self.root_path), "Root directory should be created."
        assert os.path.isdir(self.root_path), "Root path should be a directory."

    def test_version_0_created(self):
        version_path = self.workspace.resolve_path(version='0')
        assert os.path.exists(version_path), "v_0 path exists."
        assert os.path.isdir(version_path), "v_0 path is a directory."

        version_path = self.workspace.resolve_path(version='1')
        assert not os.path.exists(version_path), "v_1 does not path exist."

    def test_initial_n_versions(self):
        assert self.workspace.n_versions == 1  # v_0 only

    def test_copy_from_template_dir(self):
        file_contents = {self.workspace.resolve_path(k, version='0'): v for k, v in self.template_contents.items()}
        check_files_with_contents_exist(file_contents)

    def test_save_to_file_to_root(self):
        filename = 'root.txt'
        file_content = 'this is the root dir'
        self.workspace.save_to_file('this is the root dir', 'root.txt')
        abs_path = self.workspace.resolve_path('root.txt')

        assert os.path.exists(abs_path)
        with open(abs_path, 'r') as f:
            assert f.read() == file_content, f'File content should be "{file_content}"'

    def test_save_to_file_to_version(self):
        filename = 'hello.txt'
        file_content = 'hello world v1'
        self.workspace.save_to_file('hello world v1', 'hello.txt', version='1')
        abs_path = self.workspace.resolve_path('hello.txt', version='1')

        assert os.path.exists(abs_path)
        with open(abs_path, 'r') as f:
            assert f.read() == file_content, f'File content should be "{file_content}"'

    def test_create_version_default(self):
        version = self.workspace.create_version()
        version_path = self.workspace.resolve_path(version=version)
        assert os.path.exists(version_path)

        file_contents = {self.workspace.resolve_path(k, version=version): v for k, v in self.template_contents.items()}
        check_files_with_contents_exist(file_contents)

        assert version in self.workspace.version_infos
        assert len(self.workspace.version_infos) == self.workspace.n_versions

    def test_create_version_from_path(self):
        version = self.workspace.create_version(from_path=self.template_dir)
        version_path = self.workspace.resolve_path(version=version)
        assert os.path.exists(version_path)

        file_contents = {self.workspace.resolve_path(k, version=version): v for k, v in self.template_contents.items()}
        check_files_with_contents_exist(file_contents)

        assert version in self.workspace.version_infos
        assert len(self.workspace.version_infos) == self.workspace.n_versions

    def test_create_version_from_version(self):
        parent_version = '1'
        version = self.workspace.create_version(from_version=parent_version)
        version_path = self.workspace.resolve_path(version=version)
        assert os.path.exists(version_path)

        metadata_path = self.workspace.resolve_path('meta.json', version=version)
        assert os.path.exists(metadata_path)

        with open(metadata_path, 'r') as f:
            metadata = json.loads(f.read())
            assert metadata['parent'] == parent_version
            assert 'created_at' in metadata

        rel_file_contents = {**self.template_contents, 'hello.txt': 'hello world v1'}

        file_contents = {self.workspace.resolve_path(k, version=version): v for k, v in rel_file_contents.items()}
        check_files_with_contents_exist(file_contents)

        assert version in self.workspace.version_infos
        assert len(self.workspace.version_infos) == self.workspace.n_versions

        # @todo: Check children pointer created corrected
        parent_metadata_path = self.workspace.resolve_path('meta.json', version=parent_version)
        assert os.path.exists(parent_metadata_path)
        with open(parent_metadata_path, 'r') as f:
            parent_metadata = json.loads(f.read())
            assert parent_metadata['children'] == [version]

    def test_ls(self):
        contents = self.workspace.ls()
        assert set(contents) == set([*[f'v_{i}' for i in range(self.workspace.n_versions)], 'root.txt']), (
            'Unexpected contents.'
        )

    def test_view_root(self):
        view = self.workspace.view()
        assert view == '# root.txt\nthis is the root dir\n'

    def test_view_paths_from_root(self):
        view = self.workspace.view(paths=['root.txt', 'v_1/hello.txt'])
        assert view == '\n'.join([
            '# root.txt\nthis is the root dir\n',
            '# v_1/hello.txt\nhello world v1\n',
        ])

    def test_view_version(self):
        view = self.workspace.view(version='2')
        assert view == "# test.txt\nThese are the droids you are looking for.\n"

    def test_view_paths_from_version(self):
        view = self.workspace.view(paths='test.txt', version='1')
        assert view == "# test.txt\nThese are the droids you are looking for.\n"

    def test_view_version_recursive(self):
        view = self.workspace.view(version='2', recursive=True)
        assert view == '\n'.join([
            "# test.txt\nThese are the droids you are looking for.\n",
            "# data/test_data.csv\nname,role\nnikola tesla,scientist\n\n"
        ])

    def test_view_missing_path(self):
        with pytest.raises(ValueError):
            view = self.workspace.view(paths='missing.txt')

    def test_get_top_k_versions(self):
        versions = ['1', '2']
        scores = [1.0, 2.0]
        for version, score in zip(versions, scores):
            res = dict(metrics={'score': score})
            self.workspace.save_to_file(json.dumps(res), 'results.json', version=version)

        top_k = [x.version for x in self.workspace.get_top_k_versions('score', k=2)]

        assert top_k == ['2', '1']

        top_k = [x.version for x in self.workspace.get_top_k_versions('score', k=1)]
        assert top_k == ['2']

        top_k = [x.version for x in self.workspace.get_top_k_versions('score')]
        assert top_k == ['2'] 

        top_k = [x.version for x in self.workspace.get_top_k_versions('score', from_versions=['1'])]
        assert top_k == ['1']

    def test_version_reload_on_save_to_file(self):
        self.workspace.save_to_file(json.dumps(dict(metrics={'score': 5.0})), 'results.json', version='1')
        assert self.workspace.version_infos['1'].results == {'metrics': {'score': 5.0}}

    def test_mark_as_buggy_from_version(self):
        stable_version = self.workspace.create_version()

        buggy_version_one = self.workspace.create_version(from_version=stable_version)
        self.workspace.mark_as_buggy_from_version(version=buggy_version_one, from_version=stable_version)

        buggy_version_two = self.workspace.create_version(from_version=buggy_version_one)
        self.workspace.mark_as_buggy_from_version(version=buggy_version_two, from_version=buggy_version_one)

        assert self.workspace.version_infos[buggy_version_one].bug_depth == 1
        assert self.workspace.version_infos[buggy_version_one].stable_ancestor_version == stable_version

        assert self.workspace.version_infos[buggy_version_two].bug_depth == 2
        assert self.workspace.version_infos[buggy_version_two].stable_ancestor_version == stable_version


class TestWorkspaceViewHistory:
    """
    A test class that shares a single Workspace instance across all methods.
    """
    @pytest.fixture(autouse=True, scope="class")
    def _setup_class(self, workspace_factory):
        template_contents = {
            "test.txt": "These are the droids you are looking for.",
            "data/test_data.csv": "name,role\nnikola tesla,scientist\n"
        }
        type(self).template_contents = template_contents

        (
            type(self).workspace, 
            type(self).root_path, 
            type(self).template_dir
        ) = workspace_factory(template_contents)

        # Set up history
        #
        # [v_1]
        #   |
        # [v_2]--bug--[v_4]--bug--[v_5]--bug--[v_6]--[v_7]
        #   |
        # [v_3]
        #
        self.workspace.create_version(from_version='0')  # v_1
        self.workspace.create_version(from_version='1')  # v_2
        self.workspace.create_version(from_version='2')  # v_3

        self.workspace.create_version(from_version='2')  # v_4
        self.workspace.create_version(from_version='4')  # v_5
        self.workspace.create_version(from_version='5')  # v_6
        self.workspace.create_version(from_version='6')  # v_7

        self.workspace.mark_as_buggy_from_version(version='4', from_version='2')
        self.workspace.mark_as_buggy_from_version(version='5', from_version='4')
        self.workspace.mark_as_buggy_from_version(version='6', from_version='5')

    def test_view_history_default_only_good(self):
        history = self.workspace.view_history(as_string=False)
        assert len(history) == 5  # v_0, v_1, v_2, v_3, v_7 are good

    def test_view_history_max_len(self):
        history = self.workspace.view_history(as_string=False, max_len=1)
        assert len(history) == 1

    def test_view_history_incl_all(self):
        history = self.workspace.view_history(as_string=False, incl_buggy_versions=True)
        assert len(history) == self.workspace.n_versions

    def test_view_history_incl_buggy_versions_only(self):
        history = self.workspace.view_history(as_string=False, incl_good_versions=False, incl_buggy_versions=True)
        assert len(history) == 3

    # Test from_version conditions for ancestors and descendents
    def test_view_history_from_version_default_ancestors_only(self):
        history = self.workspace.view_history(as_string=False, from_version='3')
        assert len(history) == 3
        assert [info.version for info in history] == ['2', '1', '0']

    def test_view_history_from_version_ancestors_only_depth_one(self):
        history = self.workspace.view_history(as_string=False, from_version='3', ancestor_depth=1)
        assert len(history) == 1
        assert [info.version for info in history] == ['2']

    def test_view_history_from_version_incl_descendents_only(self):
        history = self.workspace.view_history(
            as_string=False, from_version='2', incl_ancestors=False, incl_descendents=True
        )
        assert len(history) == 2
        assert [info.version for info in history] == ['7', '3']

    def test_view_history_from_version_incl_descendents_only_depth_one(self):
        history = self.workspace.view_history(
            as_string=False, from_version='2', incl_ancestors=False, incl_descendents=True, descendent_depth=1
        )
        assert len(history) == 1
        assert [info.version for info in history] == ['3']

    def test_view_history_from_version_incl_descendents_incl_buggy_versions_only(self):
        history = self.workspace.view_history(
            as_string=False,
            from_version='2',
            incl_ancestors=False,
            incl_descendents=True,
            incl_buggy_versions=True,
            incl_good_versions=False,
        )
        assert len(history) == 3
        assert [info.version for info in history] == ['6', '5', '4']

    def test_view_history_from_version_incl_descendents_incl_buggy_versions_only_depth_one(self):
        history = self.workspace.view_history(
            as_string=False,
            from_version='2',
            incl_ancestors=False,
            incl_descendents=True,
            incl_buggy_versions=True,
            incl_good_versions=False,
            descendent_depth=1
        )
        assert len(history) == 1
        assert [info.version for info in history] == ['4']

    def test_view_history_from_version_all(self):
        history = self.workspace.view_history(
            as_string=False,
            from_version='4',
            incl_ancestors=True,
            incl_descendents=True,
            incl_buggy_versions=True,
        )
        assert len(history) == 6
        assert [info.version for info in history] == ['7', '6', '5', '2', '1', '0']

    def test_get_buggy_versions_all(self):
        infos = self.workspace.get_buggy_versions(is_leaf=False, max_bug_depth=None)
        assert len(infos) == 3
        assert set([info.version for info in infos]) == {'4', '5', '6'}

    def test_get_buggy_versions_is_leaf(self):
        infos = self.workspace.get_buggy_versions(is_leaf=True, max_bug_depth=None)
        assert len(infos) == 0

    def test_get_buggy_versions_all_max_bug_depth(self):
        infos = self.workspace.get_buggy_versions(is_leaf=False, max_bug_depth=2)
        assert len(infos) == 2
        assert set([info.version for info in infos]) == {'4', '5'}

    def test_get_buggy_versions_is_leaf_max_bug_depth(self):
        infos = self.workspace.get_buggy_versions(is_leaf=True, max_bug_depth=2)
        assert len(infos) == 0

    def test_get_good_versions(self):
        infos = self.workspace.get_good_versions()
        assert len(infos) == 5
        assert set([info.version for info in infos]) == {'1', '2', '3', '7', '0'}


class TestWorkspaceDeleteVersions:
    """
    A test class that shares a single Workspace instance across all methods.
    """
    @pytest.fixture(autouse=True, scope="class")
    def _setup_class(self, workspace_factory):
        template_contents = {
            "test.txt": "These are the droids you are looking for.",
            "data/test_data.csv": "name,role\nnikola tesla,scientist\n"
        }
        type(self).template_contents = template_contents

        (
            type(self).workspace, 
            type(self).root_path, 
            type(self).template_dir
        ) = workspace_factory(template_contents)

        # Event history:
        #
        # [v_1]
        #   |
        # [v_2] - No result, later delete
        #   |
        # [v_3]
        #   |
        # [v_4]
        #
        self.workspace.create_version(from_version='0')  # v_1
        self.workspace.create_version(from_version='1')  # v_2
        self.workspace.create_version(from_version='2')  # v_3
        self.workspace.create_version(from_version='3')  # v_4

        self.workspace.save_to_file(
            json.dumps(dict(status='COMPLETED')), 'results.json', version='1'
        )
        self.workspace.save_to_file(
            json.dumps(dict(status='COMPLETED')), 'results.json', version='2'
        )

    def test_get_completed_versions(self):
        infos = self.workspace.get_completed_versions()
        assert len(infos) == 2
        assert set([info.version for info in infos]) == {'1', '2'}

    def test_delete_version(self):
        self.workspace.delete_version(version='3')

        assert len(self.workspace.version_infos) == 3

        infos = self.workspace.get_completed_versions()
        assert len(infos) == 2
        assert set([info.version for info in infos]) == {'1', '2'}

        assert self.workspace.n_versions == 3
        assert self.workspace.max_version == 2
        assert self.workspace.version_infos['2'].children == []

            


