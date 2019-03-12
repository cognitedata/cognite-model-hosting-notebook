import enum
import re


def _should_include_code_cell(cell):
    include_by_tag = "metadata" in cell and "tags" in cell["metadata"] and "model" in cell["metadata"]["tags"]
    include_by_comment = bool(re.fullmatch(r"# *!model *\n?", cell["source"][0]))

    return include_by_tag or include_by_comment


class InvalidCodeFormat(Exception):
    """Raised if there is a mistake in how you have defined your model within the notebook."""

    pass


def extract_source_code(notebook):
    source_code = []
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code" and cell["source"]:
            if _should_include_code_cell(cell):
                source_code.append("# !notebook-cell\n")
                source_code.extend(cell["source"] + ["\n\n\n"])
    return "".join(source_code[:-1] + ["\n"])


def _get_function_signature(source_code, function_name):
    parameter_signatures = []
    for line in source_code.split("\n"):
        match = re.fullmatch(r"def {}\(([^)]*)\): *".format(function_name), line)
        if match:
            parameter_signatures.append(match.group(1))

    if not parameter_signatures:
        return None
    if len(parameter_signatures) > 1:
        raise InvalidCodeFormat("Multiple definitions of {}()".format(function_name))

    return parameter_signatures[0]


def _source_code_has_function(source_code, name, parameter_pattern, parameter_error_msg):
    parameters = _get_function_signature(source_code, name)
    if parameters is None:
        return False

    if re.fullmatch(parameter_pattern, parameters):
        return True
    raise InvalidCodeFormat(parameter_error_msg)


def _source_code_has_train(source_code):
    return _source_code_has_function(
        source_code,
        name="train_model",
        parameter_pattern=r"open_artifact(,.*)?",
        parameter_error_msg="train_model() must have `open_artifact` as first parameter (additional user specified parameters are also allowed)",
    )


def _source_code_has_load_and_predict(source_code):
    has_load = _source_code_has_function(
        source_code, "load_model", r"open_artifact", "load_model() must have `open_artifact` as the only parameter"
    )
    if has_load:
        has_predict = _source_code_has_function(
            source_code,
            name="predict",
            parameter_pattern=r"model, instance(,.*)?",
            parameter_error_msg=(
                "predict() must have `model` as first parameter and `instance` as second parameter "
                "(additional user specified parameters are also allowed) when there's a load_model() function"
            ),
        )
    else:
        has_predict = _source_code_has_function(
            source_code,
            name="predict",
            parameter_pattern=r"instance(,.*)?",
            parameter_error_msg=(
                "predict() must have `instance` as first parameter"
                "(additional user specified parameters are also allowed) when there's no load_model() function"
            ),
        )

    if has_load and not has_predict:
        raise InvalidCodeFormat("A load_model() function was specified, but not a predict() function")

    return has_load, has_predict


def _get_function_parameters(source_code, function_name):
    parameters = _get_function_signature(source_code, function_name)
    assert parameters is not None, "The function `{}` is not defined".format(function_name)
    parameters = parameters.split(",")
    parameters = [p.strip() for p in parameters]
    if len(parameters) != len(set(parameters)):
        raise InvalidCodeFormat("Duplicate parameters `` in {}()".format(function_name))
    return parameters


def _get_user_defined_predict_parameters(source_code):
    parameters = _get_function_parameters(source_code, "predict")
    return parameters[parameters.index("instance") + 1 :]


def _get_user_defined_train_parameters(source_code):
    parameters = _get_function_parameters(source_code, "train_model")
    return parameters[1:]


_glue_code_start = "# !auto-generated\nclass Model:"
_glue_code_constructor_with_state = "    def __init__(self, model):\n        self._model = model\n"
_glue_code_train = "    @staticmethod\n    def train(open_artifact, {}):\n        train_model(open_artifact, {})\n"
_glue_code_load = "    @staticmethod\n    def load(open_artifact):\n        return Model(load_model(open_artifact))\n"
_glue_code_stateless_load = "    @staticmethod\n    def load(open_artifact):\n        return Model()\n"
_glue_code_predict = "    def predict(self, instance, {}):\n        return predict(self._model, instance, {})\n"
_glue_code_stateless_predict = "    def predict(self, instance, {}):\n        return predict(instance, {})\n"


class AvailableOperations(enum.Enum):
    PREDICT = 1
    TRAIN = 2
    PREDICT_TRAIN = 3


def get_model_file_content(source_code, available_operations):
    has_train = _source_code_has_train(source_code)
    has_load, has_predict = _source_code_has_load_and_predict(source_code)

    should_have_train = available_operations in [AvailableOperations.TRAIN, AvailableOperations.PREDICT_TRAIN]
    should_have_predict = available_operations in [AvailableOperations.PREDICT, AvailableOperations.PREDICT_TRAIN]

    if should_have_train and not has_train:
        raise InvalidCodeFormat("Missing required train_model() function")
    if should_have_predict and not has_predict:
        raise InvalidCodeFormat("Missing required predict() functions")
    if not should_have_train and has_train:
        raise InvalidCodeFormat(
            "A train_model() function should not be defined when no training is performed in Model Hosting"
        )
    if not should_have_predict and has_predict:
        raise InvalidCodeFormat(
            "A predict() function should not be defined when no prediction is performed in Model Hosting"
        )

    content = [source_code + "\n"]
    content.append(_glue_code_start)
    if has_load:
        content.append(_glue_code_constructor_with_state)

    if has_train:
        user_defined_parameters = _get_user_defined_train_parameters(source_code)
        user_defined_parameters = ", ".join(user_defined_parameters)
        content.append(_glue_code_train.format(user_defined_parameters, user_defined_parameters))

    if has_predict:
        if has_load:
            content.append(_glue_code_load)
        else:
            content.append(_glue_code_stateless_load)

        user_defined_parameters = _get_user_defined_predict_parameters(source_code)
        user_defined_parameters = ", ".join(user_defined_parameters)
        if has_load:
            content.append(_glue_code_predict.format(user_defined_parameters, user_defined_parameters))
        else:
            content.append(_glue_code_stateless_predict.format(user_defined_parameters, user_defined_parameters))

    return "\n".join(content)
