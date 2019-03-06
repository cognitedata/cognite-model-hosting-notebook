import json
import os
import re
from shutil import rmtree
from time import sleep
from typing import Any, Dict
from urllib.parse import urljoin

from cognite.client import CogniteClient
from cognite.model_hosting.notebook._model_file import AvailableOperations, extract_source_code, get_model_file_content
from cognite.model_hosting.notebook._setup_file import extract_requirements, get_setup_file_content


def local_artifacts(model_version_name, root_dir=None):
    root_dir = root_dir or os.getenv("NOSTROBOOK_ROOT") or os.getcwd()

    def open_artifact(path, *args, **kwargs):
        path = os.path.join(root_dir, "artifacts", model_version_name, path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return open(path, *args, **kwargs)

    return open_artifact


class UnsupportedNotebookVersion(Exception):
    pass


def _read_notebook(path):
    with open(path, "r") as f:
        notebook = json.load(f)

    if notebook["nbformat"] != 4:
        raise UnsupportedNotebookVersion("Only notebook format version 4 (nbformat==4) is supported")

    return notebook


def _create_package(notebook_path, available_operations, name, description, build_dir):
    notebook = _read_notebook(notebook_path)
    requirements = extract_requirements(notebook)
    source_code = extract_source_code(notebook)

    rmtree(build_dir, ignore_errors=True)
    name_path_format = name.replace("-", "_")
    package_dir = os.path.join(build_dir, name_path_format)
    module_dir = os.path.join(package_dir, name_path_format)
    os.makedirs(module_dir, exist_ok=True)

    with open(os.path.join(package_dir, "__init__.py"), "w") as f:
        pass

    with open(os.path.join(module_dir, "__init__.py"), "w") as f:
        pass

    with open(os.path.join(package_dir, "setup.py"), "w") as f:
        f.write(get_setup_file_content(requirements, name, description))

    with open(os.path.join(module_dir, "model.py"), "w") as f:
        f.write(get_model_file_content(source_code, available_operations))


def _find_notebook_path():
    import ipykernel
    import requests
    from notebook.notebookapp import list_running_servers

    kernel_id = re.search("kernel-(.*).json", ipykernel.connect.get_connection_file()).group(1)
    servers = list_running_servers()
    for ss in servers:
        response = requests.get(urljoin(ss["url"], "api/sessions"), params={"token": ss.get("token", "")})
        for nn in json.loads(response.text):
            if nn["kernel"]["id"] == kernel_id:
                relative_path = nn["notebook"]["path"]
                return os.path.join(ss["notebook_dir"], relative_path)


def _sanitize_package_name(name):
    name = name.lower()
    name = name.replace("_", "-")
    name = re.sub(r"^[^a-z]+", "", name)
    name = re.sub(r"[^0-9a-z-]", "", name)
    return name


def _wait_on_uploaded_source_package(source_package_id, model_hosting_client):
    for i in range(10):
        sleep(0.5)
        source_package = model_hosting_client.source_packages.get_source_package(source_package_id)
        if source_package.is_uploaded:
            return
    raise TimeoutError("Uploading of source package timed out")


def deploy_model_version(
    name: str,
    model_id: int,
    runtime_version: str,
    artifacts_directory: str = None,
    description: str = None,
    metadata: Dict[str, str] = None,
    notebook_path: str = None,
    cognite_client: CogniteClient = None,
) -> int:
    notebook_path = notebook_path or _find_notebook_path()
    cognite_client = cognite_client or CogniteClient()

    package_name = _sanitize_package_name(name)
    _create_package(
        notebook_path=notebook_path,
        available_operations=AvailableOperations.PREDICT,
        name=package_name,
        description=description,
        build_dir="build",
    )
    model_hosting = cognite_client.experimental.model_hosting
    source_package = model_hosting.source_packages.build_and_upload_source_package(
        name=name,
        runtime_version=runtime_version,
        package_directory=os.path.join("build", package_name.replace("-", "_")),
        description=description,
        metadata=metadata,
    )
    _wait_on_uploaded_source_package(source_package.id, model_hosting)
    model_version = model_hosting.models.deploy_model_version(
        name=name,
        model_id=model_id,
        source_package_id=source_package.id,
        artifacts_directory=artifacts_directory,
        description=description,
        metadata=metadata,
    )

    return model_version.id


def train_and_deploy_model_version(
    name: str,
    model_id: int,
    runtime_version: str,
    description: str = None,
    metadata: Dict[str, str] = None,
    args: Dict[str, Any] = None,
    scale_tier: str = None,
    machine_type: str = None,
    notebook_path: str = None,
    cognite_client: CogniteClient = None,
):
    notebook_path = notebook_path or _find_notebook_path()
    cognite_client = cognite_client or CogniteClient()

    package_name = _sanitize_package_name(name)
    _create_package(
        notebook_path=notebook_path,
        available_operations=AvailableOperations.PREDICT_TRAIN,
        name=package_name,
        description=description,
        build_dir="build",
    )
    model_hosting = cognite_client.experimental.model_hosting
    source_package = model_hosting.source_packages.build_and_upload_source_package(
        name=name,
        runtime_version=runtime_version,
        package_directory=os.path.join("build", package_name.replace("-", "_")),
        description=description,
        metadata=metadata,
    )
    _wait_on_uploaded_source_package(source_package.id, model_hosting)
    model_version = model_hosting.models.train_and_deploy_model_version(
        name=name,
        model_id=model_id,
        source_package_id=source_package.id,
        description=description,
        metadata=metadata,
        args=args,
        scale_tier=scale_tier,
        machine_type=machine_type,
    )

    return model_version.id
