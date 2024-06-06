from os import environ
from typing import Any, Type


def get_env_or_default(environment_variable: str, default: Any, cast: Type = str):
    """Fetch an environment variable. If not found, return the default value. Cast the return with the expected type.

    This function was implemented to refactor the params file.

    Args:
        environment_variable (str): The environment variable name.
        default (Any): The default value.
        cast (Type, optional): The cast function to use. Defaults to str.

    Returns:
        Type of cast: The casted value (either environment variable or default)
    """
    return cast(environ.get(environment_variable, default))
