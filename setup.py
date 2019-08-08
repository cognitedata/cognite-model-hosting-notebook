import re

from setuptools import find_packages, setup

packages = find_packages(exclude=["tests*"])

version = re.search('^__version__\s*=\s*"(.*)"', open("cognite/model_hosting/notebook/__init__.py").read(), re.M).group(
    1
)

setup(
    name="cognite-model-hosting-notebook",
    version=version,
    description="Deploy notebooks to Cognite's model hosting environment",
    url="",  # TODO
    author="Nils Barlaug",
    author_email="nils.barlaug@cognite.com",
    packages=["cognite.model_hosting.notebook"],
    install_requires=["cognite-sdk>=1.0.4,<1.1.0", "requests"],
    python_requires=">=3.5",
)
