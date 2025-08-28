# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
from azure.cosmos import CosmosClient
from azureml.rag.utils.logging import get_logger

logger = get_logger("update_azure_cosmos_mongo_vcore")


def get_cosmosdb_client(connection_string: str) -> CosmosClient:
    """
    Initialize and return a CosmosDB client.

    Args:
    ----
        connection_string (str): Connection string to use for authentication to the CosmosDB client.

    """
    return CosmosClient.from_connection_string(connection_string)