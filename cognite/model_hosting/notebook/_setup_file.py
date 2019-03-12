import re


class InvalidRequirements(Exception):
    """Raised if your cell defining requirements is not correctly formatted."""

    pass


def _is_raw_requirement_cell(cell):
    return cell["cell_type"] == "raw" and "tags" in cell["metadata"] and "requirements" in cell["metadata"]["tags"]


def _extract_raw_requirement_cell(cell):
    requirements = cell["source"]
    requirements = [r.strip() for r in requirements]
    requirements = [r for r in requirements if r]
    return requirements


def _is_comment_requirement_cell(cell):
    return cell["cell_type"] == "code" and cell["source"] and re.fullmatch(r"# *!requirements *\n?", cell["source"][0])


def _sanity_check_requirements(requirements):
    valid_cmp_operators = ["<", "<=", "!=", "==", ">=", ">", "~=", "==="]
    for requirement in requirements:
        if not re.fullmatch("[a-z0-9][a-z0-9_\-.]*(({})[^ \n]+)?".format("|".join(valid_cmp_operators)), requirement):
            raise InvalidRequirements("Invalid format for the requirement `{}`".format(requirement))


def _extract_comment_requirement_cell(cell):
    lines = cell["source"][1:]
    requirements = []
    for line in lines:
        if not line.startswith("#") and not line.isspace():
            raise InvalidRequirements("All lines in the requirement cell must start with #")
        requirement = line[1:].strip()
        if requirement:
            requirements.append(requirement)
    return requirements


def _requirement_cells(notebook):
    requirement_cells = []
    for cell in notebook["cells"]:
        if _is_raw_requirement_cell(cell):
            requirement_cells.append(cell)
        elif _is_comment_requirement_cell(cell):
            requirement_cells.append(cell)

    return requirement_cells


def extract_requirements(notebook):
    requirement_cells = _requirement_cells(notebook)
    if len(requirement_cells) == 0:
        raise InvalidRequirements("Couldn't find any requirements")
    elif len(requirement_cells) > 1:
        raise InvalidRequirements("Only one requirement cell is allowed, but found multiple")

    cell = requirement_cells[0]
    if _is_raw_requirement_cell(cell):
        requirements = _extract_raw_requirement_cell(cell)
    elif _is_comment_requirement_cell(cell):
        requirements = _extract_comment_requirement_cell(cell)
    else:
        raise AssertionError

    _sanity_check_requirements(requirements)
    return requirements


def get_setup_file_content(requirements, name, description):
    requirements_str = "[" + ", ".join(['"{}"'.format(r) for r in requirements]) + "]"
    lines = []

    lines.append("from setuptools import find_packages, setup")
    lines.append("")
    lines.append("REQUIRED_PACKAGES = {}".format(requirements_str))
    lines.append("setup(")
    lines.append('    name="{}",'.format(name))
    lines.append('    version="1.0",')
    lines.append("    install_requires=REQUIRED_PACKAGES,")
    lines.append("    packages=find_packages(),")
    lines.append('    description="{}",'.format(description))
    lines.append(")")

    return "\n".join(lines) + "\n"
