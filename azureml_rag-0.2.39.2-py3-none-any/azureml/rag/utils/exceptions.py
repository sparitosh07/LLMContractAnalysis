# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Exception utils"""

import time
import logging
from functools import wraps
from typing import Tuple
from enum import Enum
from azureml.core.run import Run  # type: ignore
from azureml.rag.utils.logging import _logger_factory


class RunHistoryErrorCode(str, Enum):
    """Error code for updating Run History"""

    SERVICE_ERROR = "ServiceError"
    USER_ERROR = "UserError"


class RagErrorStrings:
    """Error strings."""

    INCORRECT_DEPLOYMENT_STATE = "Deployment is in failed or deleting state. Please resubmit job with a successful deployment."
    COMPLETION_MODEL_CANNOT_PULL_INFO = "Completion model LLM connection was unable to pull information"
    EMBEDDING_MODEL_CANNOT_PULL_INFO = "Embedding model LLM connection was unable to pull information"
    INVALID_ACS_NAME = (
        "Invalid acs index name provided. Index name must only contain lowercase letters, digits, "
        "dashes and underscores and cannot start or end with dashes and is limited to 128 characters."
    )


class BaseRagServiceError(Exception):
    """RAG Service Error."""

    def __init__(self, error_message: str = "Rag system error", exception: BaseException = None) -> None:
        """initialize the BaseRagServiceError"""
        self.exception = exception if exception else Exception(error_message)
        self.error_message = error_message

    def __str__(self):
        """return error message"""
        return self.error_message


class BaseRagUserError(Exception):
    """RAG User Error."""

    def __init__(self, error_message: str = "Rag user error", exception: BaseException = None) -> None:
        """initialize the BaseRagUserError"""
        self.exception = exception if exception else Exception(error_message)
        self.error_message = error_message

    def __str__(self):
        """return error message"""
        return self.error_message


class ValidationUserErrorIncorrectDeploymentState(BaseRagUserError):
    """RAG User Error - validate deployment fail because model deployment state is in failed or deleting state"""

    def __init__(self) -> None:
        """initialize the ValidationUserErrorIncorrectDeploymentState"""
        super().__init__(RagErrorStrings.INCORRECT_DEPLOYMENT_STATE)


class ValidationUserErrorCannotPullCompletionModel(BaseRagUserError):
    """RAG User Error - validate deployment fail because completion model LLM connection was unable to pull information"""

    def __init__(self) -> None:
        """initialize the ValidationUserErrorCannotPullCompletionModel"""
        super().__init__(RagErrorStrings.COMPLETION_MODEL_CANNOT_PULL_INFO)


class ValidationUserErrorCannotPullEmbeddingModel(BaseRagUserError):
    """RAG User Error - validate deployment fail because completion model LLM connection was unable to pull information"""

    def __init__(self) -> None:
        """initialize the ValidationUserErrorCannotPullEmbeddingModel"""
        super().__init__(RagErrorStrings.EMBEDDING_MODEL_CANNOT_PULL_INFO)


class ValidationUserErrorInvalidACSName(BaseRagUserError):
    """RAG User Error - validate deployment fail because Invalid acs index name provided"""

    def __init__(self, exception: BaseException) -> None:
        """initialize the ValidationUserErrorInvalidACSName"""
        super().__init__(RagErrorStrings.INVALID_ACS_NAME)
        self.exception = exception


class ExceptionMapper():
    """Exception Mapper."""

    def map_exception(self, exception: BaseException) -> Tuple[BaseException, RunHistoryErrorCode]:
        """Map exception to error code."""
        if isinstance(exception, BaseRagServiceError):
            return (exception, RunHistoryErrorCode.SERVICE_ERROR)
        elif isinstance(exception, BaseRagUserError):
            return (exception, RunHistoryErrorCode.USER_ERROR)
        return (BaseRagServiceError(exception=exception), RunHistoryErrorCode.SERVICE_ERROR)  # default to service error


def map_exceptions(func, activity_logger, *func_args, **kwargs):
    """Swallow all exceptions.

    Catch all the exceptions arising in the functions wherever used
    Categorize it as either SERVICE_ERROR or USER_ERROR
    Update RunHistory accordingly

    :param logger: The logger to be used for logging the exception raised
    :type logger: Instance of logging.logger
    """

    try:
        return func(*func_args, **kwargs)
    except Exception as e:
        # fail the run
        run = Run.get_context()
        (itepreted_exception, error_code) = ExceptionMapper().map_exception(e)
        run.fail(error_details=itepreted_exception, error_code=error_code)
        activity_logger.error(f"{error_code}: intepreted error = {itepreted_exception}, original error = {e}")

        raise e
    finally:
        if _logger_factory.appinsights:
            _logger_factory.appinsights.flush()
        time.sleep(5)
