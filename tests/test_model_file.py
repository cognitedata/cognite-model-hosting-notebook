import pytest

from cognite.model_hosting.notebook._model_file import (
    InvalidCodeFormat,
    _source_code_has_load_and_predict,
    _source_code_has_train,
    extract_source_code,
)


def test_extract_source_code():
    notebook = {
        "cells": [
            {"source": ["# !model\n", "abc\n", "def\n", "ghi"], "cell_type": "code"},
            {"source": ["this should be ignored"], "cell_type": "raw"},
            {"source": ["this should be ignored"], "cell_type": "code"},
            {"source": ["cool_code()"], "cell_type": "code", "metadata": {"tags": ["model"]}},
            {"source": [], "cell_type": "code", "metadata": {"tags": ["unknown-tag"]}},
        ]
    }
    expected_source_code = """# !notebook-cell
# !model
abc
def
ghi


# !notebook-cell
cool_code()
"""

    assert expected_source_code == extract_source_code(notebook)


class TestSourceCodeHasTrain:
    TRUE_CASES = [
        """
def train_model(open_artifact, data_spec):
    pass
""",
        """
def train_model(open_artifact):
    pass
""",
    ]
    FALSE_CASES = [
        """
def train(open_artifact, data_spec):
    pass
""",
        """
def load(open_artifact):
    pass
def predict(model, instance):
    pass
""",
    ]
    INVALID_CASES = [
        """
def train_model(data_spec):
    pass
""",
        """
def train_model(open, data_spec):
    pass
    """,
        """
def train_model():
    pass
""",
        """
def train_model(open_artifact):
    pass
def train_model(open_artifact):
    pass
""",
    ]

    @pytest.mark.parametrize("source_code", TRUE_CASES)
    def test_true(self, source_code):
        assert _source_code_has_train(source_code)

    @pytest.mark.parametrize("source_code", FALSE_CASES)
    def test_false(self, source_code):
        assert not _source_code_has_train(source_code)

    @pytest.mark.parametrize("source_code", INVALID_CASES)
    def test_invalid(self, source_code):
        with pytest.raises(InvalidCodeFormat, match="train_model"):
            _source_code_has_train(source_code)


class TestSourceCodeHasLoadAndPredict:
    VALID_CASES = [
        (
            """
def load_model(open_artifact):
    pass
def predict(model, instance):
    pass
""",
            True,
            True,
        ),
        (
            """
def load_model(open_artifact):
    pass
def predict(model, instance, some_argument):
    pass
""",
            True,
            True,
        ),
        (
            """
def predict(instance):
    pass
""",
            False,
            True,
        ),
        (
            """
def something():
    pass
""",
            False,
            False,
        ),
        (
            """
def load():
    pass
def predict_something():
    pass
""",
            False,
            False,
        ),
    ]
    INVALID_CASES = [
        """
def load_model():
    pass
def predict(model, instance):
    pass
""",
        """
def load_model(open_artifact):
    pass
def predict(model):
    pass
""",
        """
def predict(model, instance):
    pass
""",
        """
def load_model(open_artifact):
    pass
""",
        """
def load_model():
    pass
""",
        """
def load_model(open_artifact):
    pass
def predict(instance):
    pass
""",
        """
def load_model(open_artifact):
    pass
def load_model(open_artifact):
    pass
def predict(model, instance):
    pass
""",
    ]

    @pytest.mark.parametrize("source_code, should_have_load, should_have_predict", VALID_CASES)
    def test_valid(self, source_code, should_have_load, should_have_predict):
        has_load, has_predict = _source_code_has_load_and_predict(source_code)
        assert should_have_load == has_load
        assert should_have_predict == has_predict

    @pytest.mark.parametrize("source_code", INVALID_CASES)
    def test_invalid(self, source_code):
        with pytest.raises(InvalidCodeFormat, match="(load_model|predict)"):
            _source_code_has_load_and_predict(source_code)
