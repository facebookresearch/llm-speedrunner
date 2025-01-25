import os
import pytest
import shutil
from unittest.mock import patch, MagicMock

from scientst.core import ExperimentConfig, ExperimentRecord, ExperimentHistory
from scientist.core.workspace import Workspace


@pytest.fixture(scope="class")
def workspace_factory(tmp_path_factory):
    """
    Returns a factory function to create a Workspace with custom content.
    """

    def _create_workspace(template_contents: Dict[str, str] = None):
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
        tmp_dir = tmp_path_factory.mktemp("scientist_tests")

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
            track_history=True
        )

        return ws, root_path, template_dir

    return _create_workspace


def check_files_with_contents_exist(file_contents: dict[str, str]):
    for abs_path, expected_contents in file_contents.items():
        assert os.path.exists(abs_path), (
            f"File from {rel_path} should be copied to root_dir."
        )
        with open(abs_path, "r") as f:
            assert f.read() == expected_contents, (
                f"Copied file at {rel_path} content should match original."
            )

class TestNewWorkspace:
    """
    A test class that shares a single Workspace instance across all methods.
    """
    @pytest.fixture(autouse=True, scope="class")
    def _setup_class(self, workspace_fixture):
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
        version_0_path = self.workspace.resolve_path(version='0')
        assert os.path.exists(version_0_path), "v_0 path exists."
        assert os.path.isdir(version_0_path), "v_0 path is a directory."

    def test_initial_n_versions(self):
        assert self.n_versions = 0

    def test_copy_from_template_dir(self):
        file_contents = {self.resolve_path(k, version='0'): v for k, v in self.template_contents.items()}
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
        file_content = 'hello world v0'
        self.workspace.save_to_file('hello world v0', 'hello.txt', version='0')
        abs_path = self.workspace.resolve_path('hello.txt', version='0')

        assert os.path.exists(abs_path)
        with open(abs_path, 'r') as f:
            assert f.read() == file_content, f'File content should be "{file_content}"'

    def test_create_version_default(self):
        self.workspace.create_version()

        version = str(self.workspace.n_versions)
        version_path = self.workspace.resolve_path(version=version)
        assert os.path.exists(version_path)

        file_contents = {self.resolve_path(k, version=version): v for k, v in self.template_contents.items()}
        check_files_with_contents_exist(file_contents)

    def test_create_version_from_path(self):
        self.workspace.create_version(from_path='template_dir')

        version = str(self.workspace.n_versions)
        version_path = self.workspace.resolve_path(version=version)
        assert os.path.exists(version_path)

        file_contents = {self.resolve_path(k, version=version): v for k, v in self.template_contents.items()}
        check_files_with_contents_exist(file_contents)

    def test_create_version_from_version(self):
        self.workspace.create_version(from_version='0')
        version = str(self.workspace.n_versions)
        version_path = self.workspace.resolve_path(version=version)
        assert os.path.exists(version_path)

        rel_file_contents = {**self.template_contents, 'hello.txt': 'hello world v0'}

        file_contents = {self.resolve_path(k, version=version): v for k, v in rel_file_contents.items()}
        check_files_with_contents_exist(file_contents)

    def test_ls(self):
        contents = self.workspace.ls()
        assert contents == [*[f'v_{i}' for i in range(self.workspace.n_versions)], 'root.txt'], (
            'Unexpected contents.'
        )

    def test_view_root(self):
        view = self.workspace.view()
        assert view == '# root.txt\nthis is the root dir\n'

    def test_view_paths_from_root(self):
        view = self.workspace.view(paths=['root.txt', 'v_0/hello.txt'])
        assert view == '\n'.join([
            '# root.txt\nthis is the root dir\n',
            '# hello.txt\nhello world v0\n',
        ])

    def test_view_version(self):
        view = self.workspace.view(version='1')
        assert view == '\n'.join([
            "# test.txt\nThese are the droids you are looking for.",
            "# data/test_data.csv\nname,role\nnikola tesla,scientist\n"
        ])

    def test_view_paths_from_version(self):
        view = self.workspace.view(paths='test.txt', version='0')
        assert view == "# test.txt\nThese are the droids you are looking for."

    def test_view_version_recursive(self):
        view = self.workspace.view(version='1', recursive=True)
        assert view == '\n'.join([
            "# test.txt\nThese are the droids you are looking for.",
            "# data/test_data.csv\nname,role\nnikola tesla,scientist\n"
        ])

    def test_view_missing_path(self):
        with pytest.raises(ValueError):
            view = self.workspace.view(paths='missing.txt')


