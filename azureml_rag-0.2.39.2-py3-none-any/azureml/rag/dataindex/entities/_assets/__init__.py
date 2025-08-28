# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

__path__ = __import__("pkgutil").extend_path(__path__, __name__)

from azureml.rag.dataindex.entities._assets._artifacts import Data

__all__ = [
    "Data",
]
