# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
"""DataIndex configuration and operations."""

from azureml.rag.dataindex.data_index.models import build_model_protocol
from azureml.rag.dataindex.entities.data_index import CitationRegex, Data, DataIndex, Embedding, IndexSource, IndexStore
from azureml.rag.dataindex.entities._builders.data_index_func import index_data

__all__ = [
    "DataIndex",
    "IndexSource",
    "Data",
    "CitationRegex",
    "Embedding",
    "IndexStore",
    "index_data",
    "build_model_protocol",
]
