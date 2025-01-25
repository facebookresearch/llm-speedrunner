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
            (ws, root_path, cp_dir): The Workspace instance and its paths.
        """
        # Create a unique temp directory for each call
        tmp_dir = tmp_path_factory.mktemp("scientist_tests")

        root_path = tmp_dir / "workspace"
        cp_dir = tmp_dir / "cp_src"

        # Create cp_dir and populate it if cp_dir_contents is provided
        if template_contents:
            for relative_path, file_contents in template_contents.items():
                dest_path = cp_dir / relative_path
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                dest_path.write_text(file_contents)
        else:
            cp_dir.mkdirs(parents=True, exist_ok=True)
            (cp_dir / "default.txt").write_text("Default content")

        # Create the Workspace
        ws = Workspace(
            root_path=str(root_path),
            template_dir=str(cp_dir),
            track_history=True
        )

        return ws, root_path, cp_dir

    return _create_workspace


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
        ) = workspace_factory(cp_dir_contents)

    def test_root_directory_created(self):
        assert os.path.exists(self.root_path), "Root directory should be created."
        assert os.path.isdir(self.root_path), "Root path should be a directory."

    def test_version_0_created(self):
        version_0_path = os.path.join(self.root_dir, "v_0")
        assert os.path.exists(version_0_path), "v_0 path exists."
        assert os.path.isdir(version_0_path), "v_0 path is a directory."

    def test_initial_n_versions(self):
        assert self.n_versions = 0

    def test_copy_from_template_dir(self):
        for rel_path, expected_contents in self.template_contents.items():
            abs_path = self.resolve_path(rel_path, version='0')

            assert os.path.exists(abs_path), (
                f"File from {rel_path} should be copied to root_dir."
            )
            with open(abs_path, "r") as f:
                assert f.read() == expected_contents, (
                    f"Copied file at {rel_path} content should match original."
                )

    def test_save_to_file_to_root(self):
        pass

    def test_save_to_file_to_version(self):
        pass

    def test_create_version(self):
        pass

    def test_ls(self):
        pass

    def test_view_root(self):
        pass

    def test_view_version(self):
        pass

    def test_view_file(self):
        pass



class TestBranchedWorkspace:
    @pytest.fixture(autouse=True, scope="class")
    def _setup_class(self, workspace_fixture):
        type(self).workspace, type(self).root_path, type(self).cp_dir = workspace_fixture


    def test_initial_contents(self):
        pass


