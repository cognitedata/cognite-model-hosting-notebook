from collections import namedtuple

import pytest
from cognite.model_hosting.notebook._setup_file import (
    InvalidRequirements,
    _sanity_check_requirements,
    extract_requirements,
    get_setup_file_content,
)


class TestRequirementSanityCheck:
    SANE_CASES = ["numpy>1", "numpy==0.1", "numpy==1.2.3", "numpy>=1.2.3", "num-py"]
    NON_SANE_CASES = ["numpy 0.1", "numpy=1.2.3"]

    def test_sane(self):
        _sanity_check_requirements(self.SANE_CASES)

    @pytest.mark.parametrize("requirement", NON_SANE_CASES)
    def test_sane(self, requirement):
        with pytest.raises(InvalidRequirements):
            _sanity_check_requirements([requirement])


class TestExtractRequirements:
    ValidTestCase = namedtuple("ValidTestCase", ["name", "notebook", "expected_requirements"])
    VALID_TEST_CASES = [
        ValidTestCase(
            name="tag_first",
            notebook={
                "cells": [
                    {
                        "cell_type": "raw",
                        "metadata": {"tags": ["requirements"]},
                        "source": ["numpy==1.2.3\n ", "\n", "pandas", "scikit-learn==0.1.0"],
                    },
                    {"cell_type": "code", "metadata": {}, "source": ["print('hello')\n", "a = 5"]},
                ]
            },
            expected_requirements=["numpy==1.2.3", "pandas", "scikit-learn==0.1.0"],
        ),
        ValidTestCase(
            name="tag_last",
            notebook={
                "cells": [
                    {"cell_type": "code", "metadata": {}, "source": ["print('hello')\n", "a = 5"]},
                    {
                        "cell_type": "raw",
                        "metadata": {"tags": ["requirements"]},
                        "source": ["numpy==1.2.3\n ", "pandas"],
                    },
                ]
            },
            expected_requirements=["numpy==1.2.3", "pandas"],
        ),
        ValidTestCase(
            name="code_comment",
            notebook={
                "cells": [
                    {
                        "cell_type": "code",
                        "metadata": {},
                        "source": [
                            "# !requirements  \n",
                            "# numpy>=1.9.3\n ",
                            "\n",
                            "# pandas",
                            "#scikit-learn==0.1.0",
                        ],
                    },
                    {"cell_type": "code", "metadata": {}, "source": ["print('hello')\n", "a = 5"]},
                    {"cell_type": "code", "metadata": {}, "source": []},
                ]
            },
            expected_requirements=["numpy>=1.9.3", "pandas", "scikit-learn==0.1.0"],
        ),
    ]

    InvalidTestCase = namedtuple("ValidTestCase", ["name", "notebook", "exception_type", "error_msg_match"])
    INVALID_TEST_CASES = [
        InvalidTestCase(
            name="no_requirements",
            notebook={
                "cells": [
                    {"cell_type": "raw", "metadata": {}, "source": ["numpy==1.2.3\n ", "pandas"]},
                    {"cell_type": "code", "metadata": {}, "source": ["print('hello')\n", "a = 5"]},
                ]
            },
            exception_type=InvalidRequirements,
            error_msg_match="Couldn't find any requirements",
        ),
        InvalidTestCase(
            name="two_requirement_cells",
            notebook={
                "cells": [
                    {
                        "cell_type": "raw",
                        "metadata": {"tags": ["requirements"]},
                        "source": ["numpy==1.2.3\n ", "pandas"],
                    },
                    {"cell_type": "code", "metadata": {}, "source": ["print('hello')\n", "a = 5"]},
                    {"cell_type": "raw", "metadata": {"tags": ["requirements"]}, "source": ["some-package>0.1.0"]},
                ]
            },
            exception_type=InvalidRequirements,
            error_msg_match="one requirement cell",
        ),
        InvalidTestCase(
            name="invalid_code_comment",
            notebook={
                "cells": [
                    {"cell_type": "code", "metadata": {}, "source": ["# !requirements  \n", "numpy>=1.9.3"]},
                    {"cell_type": "code", "metadata": {}, "source": ["print('hello')\n", "a = 5"]},
                ]
            },
            exception_type=InvalidRequirements,
            error_msg_match="must start with #",
        ),
        InvalidTestCase(
            name="invalid_requirement_format",
            notebook={
                "cells": [
                    {
                        "cell_type": "raw",
                        "metadata": {"tags": ["requirements"]},
                        "source": ["numpy 1.2.3\n ", "pandas"],
                    },
                    {"cell_type": "code", "metadata": {}, "source": ["print('hello')\n", "a = 5"]},
                ]
            },
            exception_type=InvalidRequirements,
            error_msg_match="Invalid format",
        ),
    ]

    @pytest.mark.parametrize("name, notebook, expected_requirements", VALID_TEST_CASES)
    def test_valid(self, name, notebook, expected_requirements):
        assert expected_requirements == extract_requirements(notebook)

    @pytest.mark.parametrize("name, notebook, exception_type, error_msg_match", INVALID_TEST_CASES)
    def test_invalid(self, name, notebook, exception_type, error_msg_match):
        with pytest.raises(exception_type, match=error_msg_match):
            extract_requirements(notebook)


def test_get_setup_file_content():
    requirements = ["numpy", "pandas==1.2.3"]
    expected_content = """from setuptools import find_packages, setup

REQUIRED_PACKAGES = ["numpy", "pandas==1.2.3"]
setup(
    name="some_name",
    version="1.0",
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    description="some description",
)
"""
    content = get_setup_file_content(requirements, "some_name", "some description")
    assert expected_content == content
