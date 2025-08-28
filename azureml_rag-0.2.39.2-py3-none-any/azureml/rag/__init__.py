# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
"""AzureML Retrieval Augmented Generation (RAG) utilities."""

from azureml.rag.mlindex import MLIndex

__all__ = [
    "MLIndex",
]

"""Update the user agent with the current package version."""
try:
    import azureml._base_sdk_common.user_agent as user_agent
    import os
    current_package = __package__ or os.path.splitext(os.path.basename(__file__))[0]
    user_agent.append(current_package, "no_version")
except Exception:
    # Silently continue if user agent update fails
    pass
