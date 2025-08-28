# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
"""Constants."""


class IndexKinds(str):
    """Definition of index kinds"""

    AzureCognitiveSearch = "acs"
    FAISS = "faiss"
    Pinecone = "pinecone"
    Elasticsearch = "elasticsearch"
    Qdrant = "qdrant"
    Milvus = "milvus"
    MongoDB = "mongodb"
    Weaviate = "weaviate"
    AzureCosmosDBforMongoDBvCore = "azure_cosmos_mongo_vcore"
    AzureCosmosDBforNoSQL = "azure_cosmos_nosql"
    AzureCosmosDBforPostgreSQL = "azure_cosmos_postgresql"
