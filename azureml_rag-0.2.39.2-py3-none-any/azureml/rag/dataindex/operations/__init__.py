# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
"""DataIndex operations."""

__path__ = __import__("pkgutil").extend_path(__path__, __name__)

from azureml.rag.dataindex.operations._data_operations import DataOperations

__all__ = [
    "DataOperations",
]
