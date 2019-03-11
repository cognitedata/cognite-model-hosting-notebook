from setuptools import find_packages, setup

REQUIRED_PACKAGES = ["numpy", "pandas==1.2.3"]
setup(
    name="some_name",
    version="1.0",
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    description="some description",
)
