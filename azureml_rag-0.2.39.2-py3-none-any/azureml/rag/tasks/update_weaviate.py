# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
import json
import os
import time
import traceback
from pathlib import Path
from typing import Optional

import weaviate
import weaviate.classes.config as wc
import yaml
from azure.core.credentials import AzureKeyCredential, TokenCredential
from azureml.rag.embeddings import EmbeddingsContainer
from azureml.rag.indexes.index_stores import (
    INDEX_DELETE_FAILURE_MESSAGE_PREFIX,
    INDEX_UPLOAD_FAILURE_MESSAGE_PREFIX,
    WeaviateStore,
)
from azureml.rag.mlindex import MLIndex
from azureml.rag.utils.connections import (
    get_connection_by_id_v2,
    get_connection_credential,
    get_metadata_from_connection,
)
from azureml.rag.utils.exceptions import map_exceptions
from azureml.rag.utils.logging import (
    _logger_factory,
    enable_appinsights_logging,
    enable_stdout_logging,
    get_logger,
    safe_mlflow_start_run,
    track_activity,
)

logger = get_logger("update_weaviate")


def create_weaviate_index_sdk(
    weaviate_config: dict,
    weaviate_client,
    embeddings: Optional[EmbeddingsContainer] = None,
):
    """
    Create a Weaviate index using the weaviate-client SDK.

    Args:
    ----
        weaviate_config (dict): Weaviate configuration dictionary. Expected to contain:
            - index_spec: Weaviate index spec
            - index_name: Weaviate index name
            - field_mapping: Mappings from MLIndex fields (MLIndex.INDEX_FIELD_MAPPING_TYPES) to Weaviate fields.

        api_key (str): API key to use for authentication to the Weaviate index.
        embeddings (EmbeddingsContainer): EmbeddingsContainer to use for creating the index. If provided, the index
                                          will be configured to support vector search.

    """
    logger.info(f"Ensuring Weaviate index {weaviate_config['index_name']} exists")

    response = weaviate_client.collections.list_all(simple=True)
    collections = list(response.keys())

    index_name = weaviate_config.get("index_name", None)
    if not index_name:
        raise Exception("No index_name specified in Weaviate config.")

    if index_name not in collections:
        logger.info(
            f"Creating {index_name} Weaviate collection with dimensions {embeddings.get_embedding_dimensions()}."
        )

        embedding_field = weaviate_config[MLIndex.INDEX_FIELD_MAPPING_KEY].get("embedding", None)
        if embedding_field:
            weaviate_client.collections.create(
                name=index_name,
                vectorizer_config=[
                    # TODO: Weaviate supports multiple embedding fields. Have MLIndex support that as well.
                    wc.Configure.NamedVectors.none(
                        name=embedding_field
                    )
                ],
                properties=[
                    wc.Property(name="question", data_type=wc.DataType.TEXT)
                    for field in weaviate_config[MLIndex.INDEX_FIELD_MAPPING_KEY]
                    if field != "embedding"
                ],
            )
        else:
            # If no named embedding field specified, Weaviate will still create a default vector field behind the scene.
            weaviate_client.collections.create(
                name=index_name,
                properties=[
                    wc.Property(name=field, data_type=wc.DataType.TEXT)
                    for field in weaviate_config[MLIndex.INDEX_FIELD_MAPPING_KEY].values()
                ],
            )

        logger.info(f"Created {index_name} Weaviate index.")
    else:
        logger.info(f"Weaviate index {index_name} already exists")


def create_index_from_raw_embeddings(
    emb: EmbeddingsContainer,
    weaviate_config: dict = {},
    connection: dict = {},
    output_path: Optional[str] = None,
    credential: Optional[TokenCredential] = None,
    verbosity: int = 1,
) -> MLIndex:
    """
    Upload an EmbeddingsContainer to Weaviate and return an MLIndex.

    Args:
    ----
        emb (EmbeddingsContainer): EmbeddingsContainer to use for creating the index. If provided, the index
                                   will be configured to support vector search.
        weaviate_config (dict): Weaviate configuration dictionary. Expected to contain:
            - index_spec
            - api_key
            - index_name: Weaviate index name
            - field_mapping: Mappings from MLIndex fields (MLIndex.INDEX_FIELD_MAPPING_TYPES) to Pinecone fields.

        connection (dict): Configuration dictionary describing the type of the connection to the Pinecone index.
        output_path (str): The output path to store the MLIndex.
        credential (TokenCredential): Azure credential to use for authentication.
        verbosity (int): Defaults to 1.
            - 1: Log aggregate information about documents and IDs of deleted documents.
            - 2: Log all document_ids as they are processed.

    """
    with track_activity(
        logger, "create_index_from_raw_embeddings", custom_dimensions={"num_documents": len(emb._document_embeddings)}
    ) as activity_logger:
        logger.info("Updating Weaviate index...")

        if MLIndex.INDEX_FIELD_MAPPING_KEY not in weaviate_config:
            weaviate_config[MLIndex.INDEX_FIELD_MAPPING_KEY] = {
                "doc_id": "doc_id",
                "content": "content",
                "url": "url",
                "filename": "filepath",
                "title": "title",
                "metadata": "metadata_json_string",
            }

        logger.info(f"Using Index fields: {json.dumps(weaviate_config[MLIndex.INDEX_FIELD_MAPPING_KEY], indent=2)}")

        cluster_url = weaviate_config["index_spec"].get("cluster_url", None)
        if not cluster_url:
            raise Exception("No cluster_url specified in Weaviate index spec.")

        # connection_credential = get_connection_credential(connection, credential=credential)
        # if not isinstance(connection_credential, AzureKeyCredential):
        #     raise ValueError(
        #         f"Expected credential to Pinecone index to be an AzureKeyCredential, instead got: {type(connection_credential)}"
        #     )

        weaviate_client = weaviate.connect_to_wcs(
            cluster_url=cluster_url,
            auth_credentials=weaviate.auth.AuthApiKey(weaviate_config["api_key"]),
        )

        create_weaviate_index_sdk(weaviate_config, weaviate_client, embeddings=emb)

        weaviate_index_client = weaviate_client.collections.get(weaviate_config["index_name"])
        weaviate_index_store = WeaviateStore(weaviate_index_client)

        # Upload documents
        has_embeddings = emb and emb.kind != "none"
        logger.info(f"Has embeddings: {has_embeddings}")

        batch_size = weaviate_config.get("batch_size", 100)

        if has_embeddings:
            emb.upload_to_index(
                weaviate_index_store,
                weaviate_config,
                batch_size=batch_size,
                activity_logger=activity_logger,
                verbosity=verbosity,
            )
        else:
            logger.error("Documents do not have embeddings and therefore cannot upload to Pinecone index")
            raise RuntimeError("Failed to upload to Pinecone index since documents do not have embeddings")

        # Generate MLindex
        logger.info("Writing MLIndex yaml")

        mlindex_config = {"embeddings": emb.get_metadata()}
        mlindex_config["index"] = {
            "kind": "weaviate",
            "index": weaviate_config["index_name"],
            "field_mapping": weaviate_config[MLIndex.INDEX_FIELD_MAPPING_KEY],
        }

        # mlindex_config["index"] = {**mlindex_config["index"], **connection}

        if output_path is not None:
            output = Path(output_path)
            output.mkdir(parents=True, exist_ok=True)
            with open(output / "MLIndex", "w") as f:
                yaml.dump(mlindex_config, f)

    mlindex = MLIndex(uri=output_path, mlindex_config=mlindex_config)

    return mlindex
