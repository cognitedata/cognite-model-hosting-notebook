import json
import os
import re
from shutil import rmtree
from time import sleep
from typing import Any, Callable, Dict, Optional
from urllib.parse import urljoin

from cognite.client import CogniteClient
from cognite.model_hosting.notebook._model_file import AvailableOperations, extract_source_code, get_model_file_content
from cognite.model_hosting.notebook._setup_file import extract_requirements, get_setup_file_content


def local_artifacts(model_version_name: str, root_dir: Optional[str] = None) -> Callable:
    """Local artifacts storage.

    Returns a function which works like the builtin open(), but pointing to a directory specific to the provided model
    version name. The directory will be artifacts/<model_version_name>. By default the artifacts directory will reside
    in the current working directory.

    Args:
         model_version_name (str): The name of the model version which the artifacts belong to.
         root_dir (str, optional): The root directory where the `artifacts` directory reside.
            Defaults to the current working directory.

    Returns:
        Callable: A function/context manager which works like the builtin `open`. Let's your read and write to the
            local artifacts directory.

    Examples:
        Using local_artifacts()::

            open_artifact = local_artifacts('my-model')
            with open_artifact('my_file.txt', 'w') as f:
                f.write('A cool file\\n') # will be stored in artifacts/my-model/my_file.txt
    """
    root_dir = root_dir or os.getenv("MODEL_HOSTING_NOTEBOOK_ROOT") or os.getcwd()

    def open_artifact(path, *args, **kwargs):
        path = os.path.join(root_dir, "artifacts", model_version_name, path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return open(path, *args, **kwargs)

    return open_artifact


class UnsupportedNotebookVersion(Exception):
    """Raised if the current version of Jupyter Notebook is not supported by this integration."""

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
    artifacts_directory: Optional[str] = None,
    description: Optional[str] = None,
    metadata: Optional[Dict[str, str]] = None,
    notebook_path: Optional[str] = None,
    cognite_client: Optional[CogniteClient] = None,
) -> int:
    """Deploy the model version in the current notebook to the model hosting environment.

    Args:
        name (str): The name of the model version.
        model_id (int): Id of the model to deploy the version to.
        runtime_version (str): The model hosting runtime version to deploy the model to.
        artifacts_directory (str, optional): Path of the directory containing any artifacts you want to include
            with your deployment.
        description (str, optional): Description of this model version.
        metadata (Dict[str,str], optional): Any metadata to include about this model verison.
        notebook_path (str, optional): The path to the notebook. If omitted, the notebook you're in will be used.
        cognite_client (CogniteClient, optional): The CogniteClient instance to use for uploading the model.
            If omitted, a new instance will be created using the API key in the COGNITE_API_KEY environment variable.
    Returns:
        int: The model version id
    """
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
    description: Optional[str] = None,
    metadata: Optional[Dict[str, str]] = None,
    args: Optional[Dict[str, Any]] = None,
    scale_tier: Optional[str] = None,
    machine_type: Optional[str] = None,
    notebook_path: Optional[str] = None,
    cognite_client: Optional[CogniteClient] = None,
) -> int:
    """Train and deploy the model version in the current notebook in the model hosting environment.

    Args:
        name (str): The name of the model version.
        model_id (int): Id of the model to deploy the version to.
        runtime_version (str): The model hosting runtime version to deploy the model to.
        description (str, optional): Description of this model version.
        metadata (Dict[str,str], optional): Any metadata to incldue about this model verison.
        args (Dict[str, Any], optional): Arguments to pass to the train function defined on your model.
        scale_tier (str, optional): Scale tier to train on. Must be "CUSTOM" or "BASIC".
        machine_type (str, optional): Machine type to use. Only applicable if scale_tier is "CUSTOM".
        notebook_path (str, optional): The path to the notebook. If omitted, the notebook you're in will be used.
        cognite_client (CogniteClient, optional): The CogniteClient instance to use for uploading the model.
            If omitted, a new instance will be created using the API key in the COGNITE_API_KEY environment variable.
    Returns:
        int: The model version id
    """
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
