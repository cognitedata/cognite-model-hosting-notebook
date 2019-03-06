import json
import os
import tempfile
from collections import namedtuple
from glob import glob
from shutil import rmtree
from unittest.mock import MagicMock, patch

import pytest
from cognite.client import CogniteClient
from cognite.model_hosting.notebook.notebook import (
    AvailableOperations,
    UnsupportedNotebookVersion,
    _create_package,
    _read_notebook,
    _sanitize_package_name,
    deploy_model_version,
    local_artifacts,
    train_and_deploy_model_version,
)


@pytest.fixture(scope="session", autouse=True)
def working_dir():
    old_cwd = os.getcwd()
    os.chdir(os.path.dirname(__file__))
    yield
    os.chdir(old_cwd)


@pytest.fixture
def clean_artifacts():
    cwd_artifacts_folder = os.path.join(os.getcwd(), "artifacts")
    if os.path.exists(cwd_artifacts_folder):
        rmtree(cwd_artifacts_folder)


class TestLocalArtifacts:
    @pytest.mark.usefixtures("clean_artifacts")
    def test_default_root_dir(self):
        self.run_test(lambda name: local_artifacts(name), os.getcwd())

    def test_argument_root_dir(self):
        with tempfile.TemporaryDirectory() as dir:
            self.run_test(lambda name: local_artifacts(name, root_dir=dir), dir)

    def test_env_root_dir(self):
        with tempfile.TemporaryDirectory() as dir:
            os.environ["NOSTROBOOK_ROOT"] = dir
            self.run_test(lambda name: local_artifacts(name), dir)
            del os.environ["NOSTROBOOK_ROOT"]

    def run_test(self, _local_artifacts, root_dir):
        expected_path = os.path.join(root_dir, "artifacts", "some_model", "model.txt")
        assert not os.path.exists(expected_path)  # Make sure it doesn't already exist from last run

        open_artifact = _local_artifacts("some_model")
        with open_artifact("model.txt", "w") as f:
            f.write("123")

        open_artifact = _local_artifacts("some_model")
        with open_artifact("model.txt", "r") as f:
            assert "123" == f.read()

        assert os.path.exists(expected_path)
        with open(expected_path, "r") as f:
            assert "123" == f.read()


class TestReadNotebook:
    VALID_VERSION = {"cells": [], "nbformat": 4}
    INVALID_VERSION = {"cells": [], "nbformat": 5}

    def test_valid_version(self):
        with tempfile.TemporaryDirectory() as dir:
            path = os.path.join(dir, "notebook.ipynb")
            with open(path, "w") as f:
                json.dump(self.VALID_VERSION, f)
            assert {"cells": [], "nbformat": 4} == _read_notebook(path)

    def test_invalid_version(self):
        with tempfile.TemporaryDirectory() as dir:
            path = os.path.join(dir, "notebook.ipynb")
            with open(path, "w") as f:
                json.dump(self.INVALID_VERSION, f)
            with pytest.raises(UnsupportedNotebookVersion):
                _read_notebook(path)


@pytest.mark.parametrize(
    "example, available_operations",
    [
        ("train_predict_example", AvailableOperations.PREDICT_TRAIN),
        ("train_example", AvailableOperations.TRAIN),
        ("predict_example", AvailableOperations.PREDICT),
        ("predict_without_load_example", AvailableOperations.PREDICT),
    ],
)
def test_create_package(example, available_operations):
    _create_package(
        os.path.join(example, "notebook.ipynb"), available_operations, "some_name", "some description", "build"
    )

    expected_files = sorted([f for f in glob(os.path.join(example, "build/**"), recursive=True) if os.path.isfile(f)])
    generated_files = sorted([f for f in glob("build/**", recursive=True) if os.path.isfile(f)])
    assert len(expected_files) == len(generated_files)

    for file_path1, file_path2 in zip(expected_files, generated_files):
        assert file_path1.endswith(file_path2)
        with open(file_path1) as f1:
            with open(file_path2) as f2:
                assert f1.read() == f2.read()


@pytest.mark.parametrize(
    "name, expected_output",
    [
        ("something", "something"),
        ("some_thing", "some-thing"),
        ("sometHing", "something"),
        ("123something", "something"),
        ("s123omething", "s123omething"),
        ("-something", "something"),
        ("some-thing", "some-thing"),
        ("#some^%$^thing", "something"),
    ],
)
def test_sanitize_package_name(name, expected_output):
    assert expected_output == _sanitize_package_name(name)


class TestDeployModelVersion:
    @patch("cognite.model_hosting.notebook.notebook._create_package")
    def test_check_all_calls(self, create_package):
        cognite_client = MagicMock(CogniteClient)
        model_hosting = cognite_client.experimental.model_hosting
        source_package = namedtuple("SourcePackage", ["id"])(234)
        model_version = namedtuple("ModelVersion", ["id"])(345)
        model_hosting.source_packages.build_and_upload_source_package.return_value = source_package
        model_hosting.models.deploy_model_version.return_value = model_version

        model_version_id = deploy_model_version(
            name="123some_name",
            model_id=123,
            runtime_version="0.1",
            artifacts_directory="artifacts/some_model",
            description="some description",
            metadata={"key": "value"},
            notebook_path="path/notebook.ipynb",
            cognite_client=cognite_client,
        )
        assert 345 == model_version_id

        create_package.assert_called_once_with(
            notebook_path="path/notebook.ipynb",
            available_operations=AvailableOperations.PREDICT,
            name="some-name",
            description="some description",
            build_dir="build",
        )
        model_hosting.source_packages.build_and_upload_source_package.assert_called_once_with(
            name="123some_name",
            runtime_version="0.1",
            package_directory="build/some_name",
            description="some description",
            metadata={"key": "value"},
        )
        model_hosting.models.deploy_model_version.assert_called_once_with(
            name="123some_name",
            model_id=123,
            source_package_id=234,
            artifacts_directory="artifacts/some_model",
            description="some description",
            metadata={"key": "value"},
        )

    @patch("cognite.model_hosting.notebook.notebook._find_notebook_path", return_value="path/notebook.py")
    @patch("cognite.model_hosting.notebook.notebook._create_package")
    def test_find_notebook(self, create_package, find_notebook_path):
        cognite_client = MagicMock(CogniteClient)
        deploy_model_version(name="123some_name", model_id=123, runtime_version="0.1", cognite_client=cognite_client)

        assert "path/notebook.py" == create_package.call_args[1]["notebook_path"]

    @patch("cognite.model_hosting.notebook.notebook.CogniteClient")
    @patch("cognite.model_hosting.notebook.notebook._create_package")
    def test_default_client(self, create_package, CogniteClient_mock):
        deploy_model_version(
            name="123some_name", model_id=123, runtime_version="0.1", notebook_path="path/notebook.ipynb"
        )

        CogniteClient_mock.assert_called_once_with()


class TestTrainAndDeployModelVersion:
    @patch("cognite.model_hosting.notebook.notebook._create_package")
    def test_check_all_calls(self, create_package):
        cognite_client = MagicMock(CogniteClient)
        model_hosting = cognite_client.experimental.model_hosting
        source_package = namedtuple("SourcePackage", ["id"])(234)
        model_version = namedtuple("ModelVersion", ["id"])(345)
        model_hosting.source_packages.build_and_upload_source_package.return_value = source_package
        model_hosting.models.train_and_deploy_model_version.return_value = model_version

        model_version_id = train_and_deploy_model_version(
            name="123some_name",
            model_id=123,
            runtime_version="0.1",
            description="some description",
            metadata={"key": "value"},
            args={"data_spec": "DATA_SPEC"},
            scale_tier="SCALE_TIER",
            machine_type="MACHINE_TYPE",
            notebook_path="path/notebook.ipynb",
            cognite_client=cognite_client,
        )
        assert 345 == model_version_id

        create_package.assert_called_once_with(
            notebook_path="path/notebook.ipynb",
            available_operations=AvailableOperations.PREDICT_TRAIN,
            name="some-name",
            description="some description",
            build_dir="build",
        )
        model_hosting.source_packages.build_and_upload_source_package.assert_called_once_with(
            name="123some_name",
            runtime_version="0.1",
            package_directory="build/some_name",
            description="some description",
            metadata={"key": "value"},
        )
        model_hosting.models.train_and_deploy_model_version.assert_called_once_with(
            name="123some_name",
            model_id=123,
            source_package_id=234,
            description="some description",
            metadata={"key": "value"},
            args={"data_spec": "DATA_SPEC"},
            scale_tier="SCALE_TIER",
            machine_type="MACHINE_TYPE",
        )

    @patch("cognite.model_hosting.notebook.notebook._find_notebook_path", return_value="path/notebook.py")
    @patch("cognite.model_hosting.notebook.notebook._create_package")
    def test_find_notebook(self, create_package, find_notebook_path):
        cognite_client = MagicMock(CogniteClient)
        train_and_deploy_model_version(
            name="123some_name", model_id=123, runtime_version="0.1", cognite_client=cognite_client
        )

        assert "path/notebook.py" == create_package.call_args[1]["notebook_path"]

    @patch("cognite.model_hosting.notebook.notebook.CogniteClient")
    @patch("cognite.model_hosting.notebook.notebook._create_package")
    def test_default_client(self, create_package, CogniteClient_mock):
        train_and_deploy_model_version(
            name="123some_name", model_id=123, runtime_version="0.1", notebook_path="path/notebook.ipynb"
        )

        CogniteClient_mock.assert_called_once_with()
