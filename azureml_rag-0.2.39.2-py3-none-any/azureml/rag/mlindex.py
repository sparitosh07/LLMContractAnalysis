# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
"""MLIndex class for interacting with MLIndex assets."""

import os
import tempfile
import uuid
from datetime import datetime
from importlib.util import find_spec
from pathlib import Path
from typing import Any, ClassVar, Dict, Iterator, Optional, Union

import yaml
from azure.core.credentials import TokenCredential
from packaging import version as pkg_version

from azureml.rag._constants import IndexKinds
from azureml.rag.documents import Document, DocumentChunksIterator
from azureml.rag.embeddings import EmbeddingsContainer
from azureml.rag.utils.connections import (
    Connection,
    get_connection_by_id_v2,
    get_connection_credential,
    get_id_from_connection,
    get_metadata_from_connection,
    get_target_from_connection,
)
from azureml.rag.utils.constants import ACS_API_VERSION
from azureml.rag.utils.logging import (
    get_logger,
    langchain_version,
    packages_versions_for_compatibility,
    track_activity,
    version,
)

try:
    from langchain.document_loaders.base import BaseLoader
except ImportError:
    BaseLoader = Iterator[Document]

logger = get_logger("mlindex")


class MLIndex:
    """MLIndex class for interacting with MLIndex assets."""

    INDEX_FIELD_MAPPING_KEY: ClassVar[str] = "field_mapping"
    INDEX_FIELD_MAPPING_TYPES: ClassVar[Dict[str, str]] = {
        "content": "Raw data content of indexed document",
        "embedding": "Embedding of indexed document content",
        "metadata": "Metadata of indexed document, must be a JSON string",
        "filename": "Filename of indexed document, relative to data source root",
        "title": "Title of indexed document",
        "url": "User facing citation URL for indexed document",
    }

    base_uri: str
    index_config: dict
    embeddings_config: dict

    def __init__(self, uri: Optional[Union[str, Path, object]] = None, mlindex_config: Optional[dict] = None):
        """
        Initialize MLIndex from a URI or AzureML Data Asset.

        Args:
        ----
            uri: URI to MLIndex asset folder (remote or local)
            mlindex_config: MLIndex config dictionary
            credential: Credential to use for talking to Azure resources

        """
        with track_activity(logger, "MLIndex.__init__") as activity_logger:
            if uri is not None:
                # if uri is not str or Path, then assume given AzureML Data Asset
                uri = str(uri) if isinstance(uri, (str, Path)) else uri.path

                if find_spec("fsspec") is None:
                    raise ValueError(
                        "fsspec python package not installed. Please install it with `pip install fsspec`."
                    )

                if find_spec("azureml.fsspec") is None:
                    logger.warning(
                        "azureml-fsspec python package not installed. "
                        "Loading from remote filesystems supported by AzureML will not work. "
                        "Please install it with `pip install azureml-fsspec`."
                    )

                self.base_uri = uri

                mlindex_config = None
                uri = uri.rstrip("/")
                mlindex_uri = f"{uri}/MLIndex" if not uri.endswith("MLIndex") else uri
                try:
                    import fsspec

                    mlindex_file = fsspec.open(mlindex_uri, "r")
                    if hasattr(mlindex_file.fs, "_path"):
                        # File on azureml filesystem has path relative to container root
                        # so need to get underlying fs path
                        self.base_uri = mlindex_file.fs._path.split("/MLIndex")[0]
                    else:
                        self.base_uri = mlindex_file.path.split("/MLIndex")[0]

                    with mlindex_file as f:
                        mlindex_config = yaml.safe_load(f)
                except Exception as e:
                    raise ValueError(f"Could not find MLIndex: {e}") from e
            elif mlindex_config is None:
                raise ValueError("Must provide either uri or mlindex_config")
            else:
                self.base_uri = None

            self.index_config = mlindex_config.get("index", {})
            if self.index_config is None:
                raise ValueError("Could not find index config in MLIndex yaml")
            activity_logger.activity_info["index_kind"] = self.index_config.get("kind", "none")
            self.embeddings_config = mlindex_config.get("embeddings", {})
            if self.embeddings_config is None:
                raise ValueError("Could not find embeddings config in MLIndex yaml")
            activity_logger.activity_info["embeddings_kind"] = self.embeddings_config.get("kind", "none")
            activity_logger.activity_info["embeddings_api_type"] = self.embeddings_config.get("api_type", "none")

    @property
    def name(self) -> str:
        """Returns the name of the MLIndex."""
        return self.index_config.get("name", self.index_config.get("index", ""))

    @name.setter
    def name(self, value: str):
        """Sets the name of the MLIndex."""
        self.index_config["name"] = value

    @property
    def description(self) -> str:
        """Returns the description of the MLIndex."""
        return self.index_config.get("description", "")

    @description.setter
    def description(self, value: str):
        """Sets the description of the MLIndex."""
        self.index_config["description"] = value

    def get_langchain_embeddings(self, credential: Optional[TokenCredential] = None):
        """Get the LangChainEmbeddings from the MLIndex."""
        embeddings = EmbeddingsContainer.from_metadata(self.embeddings_config.copy())

        return embeddings.as_langchain_embeddings(credential=credential)

    def as_langchain_vectorstore(self, credential: Optional[TokenCredential] = None):
        """Converts MLIndex to a retriever object that can be used with langchain, may download files."""
        with track_activity(logger, "MLIndex.as_langchain_vectorstore") as activity_logger:
            index_kind = self.index_config.get("kind", "none")

            activity_logger.activity_info["index_kind"] = index_kind
            activity_logger.activity_info["embeddings_kind"] = self.embeddings_config.get("kind", "none")
            activity_logger.activity_info["embeddings_api_type"] = self.embeddings_config.get("api_type", "none")

            langchain_pkg_version = pkg_version.parse(langchain_version)

            if index_kind == IndexKinds.AzureCognitiveSearch:
                from azureml.rag.indexes.azure_search import import_azure_search_or_so_help_me

                import_azure_search_or_so_help_me()

                if self.index_config.get("field_mapping", {}).get("embedding", None) is None:
                    raise ValueError(
                        "field_mapping.embedding must be set in MLIndex config for acs index, try `.as_langchain_retriever()` instead."
                    )

                logger.info(f"Get ACS credential for with credential:{type(credential)}.")
                try:
                    connection_credential = get_connection_credential(self.index_config, credential=credential)
                except Exception as e:
                    # azure.ai.generative has workflow where env vars are set before doing stuff.
                    # AZURE_SEARCH_KEY is newer key,
                    # AZURE_AI_SEARCH_KEY is new key,
                    # but fall back to AZURE_COGNITIVE_SEARCH_KEY for backward compat.

                    # List of environment variable keys in order of preference
                    env_keys = ["AZURE_SEARCH_KEY", "AZURE_AI_SEARCH_KEY", "AZURE_COGNITIVE_SEARCH_KEY"]
                    # Find the first available key
                    search_key = next((os.environ[key] for key in env_keys if key in os.environ), None)
                    if search_key:
                        from azure.core.credentials import AzureKeyCredential

                        logger.warning(f"Failed to get credential for ACS with {e}, falling back to env vars.")
                        connection_credential = AzureKeyCredential(search_key)
                    else:
                        raise e

                azure_search_documents_version = packages_versions_for_compatibility["azure-search-documents"]
                if langchain_pkg_version >= pkg_version.parse("0.1.00"):
                    from langchain_community.vectorstores.azuresearch import AzureSearch

                    from azureml.rag.utils.acs import AzureSearchModuleSettings, AzureSearchProxy

                    fields_id = self.index_config.get("field_mapping", {}).get("id", "id")
                    fields_content = self.index_config.get("field_mapping", {}).get("content", "content")
                    fields_content_vector = self.index_config.get("field_mapping", {}).get(
                        "embedding", "content_vector_open_ai"
                    )
                    fields_metadata = self.index_config.get("field_mapping", {}).get("metadata", "meta_json_string")

                    from azure.core.credentials import AzureKeyCredential

                    endpoint = self.index_config.get("endpoint", None)
                    if not endpoint:
                        endpoint = get_target_from_connection(
                            get_connection_by_id_v2(self.index_config["connection"]["id"], credential=credential)
                        )

                    azuresearch_instance = AzureSearch(
                        azure_search_endpoint=endpoint,
                        azure_search_key=connection_credential.key
                        if isinstance(connection_credential, AzureKeyCredential)
                        else None,
                        index_name=self.index_config.get("index"),
                        embedding_function=self.get_langchain_embeddings(credential=credential).embed_query,
                        search_type="hybrid",
                        semantic_configuration_name=self.index_config.get(
                            "semantic_configuration_name", "azureml-default"
                        ),
                        user_agent=f"azureml-rag=={version}/mlindex,langchain=={langchain_version}",
                    )

                    module_settings = AzureSearchModuleSettings(
                        FIELDS_ID=fields_id,
                        FIELDS_CONTENT=fields_content,
                        FIELDS_CONTENT_VECTOR=fields_content_vector,
                        FIELDS_METADATA=fields_metadata,
                    )

                    return AzureSearchProxy(azuresearch_instance=azuresearch_instance, module_settings=module_settings)
                else:
                    from azureml.rag.langchain.acs import AzureCognitiveSearchVectorStore

                    logger.warning(
                        f"azure-search-documents=={azure_search_documents_version} not compatible langchain.vectorstores.azuresearch yet, using REST client based VectorStore."
                    )

                    return AzureCognitiveSearchVectorStore(
                        index_name=self.index_config.get("index"),
                        endpoint=self.index_config.get(
                            "endpoint",
                            get_target_from_connection(
                                get_connection_by_id_v2(self.index_config["connection"]["id"], credential=credential)
                            ),
                        ),
                        embeddings=self.get_langchain_embeddings(credential=credential),
                        field_mapping=self.index_config.get("field_mapping", {}),
                        credential=connection_credential,
                    )
            elif index_kind == IndexKinds.FAISS:
                from fsspec.core import url_to_fs

                store = None
                engine = self.index_config.get("engine")
                if engine == "langchain.vectorstores.FAISS":
                    embeddings = EmbeddingsContainer.from_metadata(
                        self.embeddings_config.copy()
                    ).as_langchain_embeddings(credential=credential)

                    # langchain fix https://github.com/langchain-ai/langchain/pull/10823 released in 0.0.318
                    if langchain_pkg_version >= pkg_version.parse("0.0.318"):
                        embeddings = embeddings.embed_query

                    fs, uri = url_to_fs(self.index_config.get("path", self.base_uri))

                    with tempfile.TemporaryDirectory() as tmpdir:
                        fs.download(f"{uri.rstrip('/')}/index.pkl", f"{tmpdir!s}")
                        fs.download(f"{uri.rstrip('/')}/index.faiss", f"{tmpdir!s}")

                        try:
                            if langchain_pkg_version >= pkg_version.parse("0.1.0"):
                                from langchain_community.vectorstores import FAISS

                                store = FAISS.load_local(str(tmpdir), embeddings, allow_dangerous_deserialization=True)
                            else:
                                from langchain.vectorstores import FAISS

                                store = FAISS.load_local(str(tmpdir), embeddings)
                        except Exception as e:
                            logger.warning(
                                f"Failed to load FAISS Index using installed version of langchain, retrying with vendored FAISS VectorStore.\n{e}"
                            )
                            from azureml.rag.langchain.vendor.vectorstores.faiss import FAISS

                            store = FAISS.load_local(str(tmpdir), embeddings)
                elif engine.endswith("indexes.faiss.FaissAndDocStore"):
                    from azureml.rag.indexes.faiss import FaissAndDocStore

                    error_fmt_str = """Failed to import langchain faiss bridge module with: {e}\n"
                        This could be due to an incompatible change in langchain since this bridge was implemented.
                        If you understand what has changed you could implement your own wrapper of azure.ai.tools.mlindex.indexes.faiss.FaissAndDocStore.
                        """
                    try:
                        from azureml.rag.langchain.faiss import azureml_faiss_as_langchain_faiss
                    except Exception as e:
                        logger.warning(error_fmt_str.format(e=e))
                        azureml_faiss_as_langchain_faiss = None

                    embeddings = EmbeddingsContainer.from_metadata(
                        self.embeddings_config.copy()
                    ).as_langchain_embeddings(credential=credential)

                    store = FaissAndDocStore.load(self.index_config.get("path", self.base_uri), embeddings.embed_query)
                    if azureml_faiss_as_langchain_faiss is not None:
                        try:
                            store = azureml_faiss_as_langchain_faiss(
                                FaissAndDocStore.load(
                                    self.index_config.get("path", self.base_uri), embeddings.embed_query
                                )
                            )
                        except Exception as e:
                            logger.error(error_fmt_str.format(e=e))
                            raise
                else:
                    raise ValueError(f"Unknown engine: {engine}")
                return store
            elif index_kind == IndexKinds.Elasticsearch:
                try:
                    from azure.core.credentials import AzureKeyCredential
                    from langchain_community.vectorstores import ElasticsearchStore

                    connection_credential = get_connection_credential(self.index_config, credential=credential)
                    if not isinstance(connection_credential, AzureKeyCredential):
                        raise ValueError(
                            f"Expected credential to Elasticsearch index to be an AzureKeyCredential, instead got: {type(connection_credential)}"
                        )
                    # Get the connection credential to connect to elasticSearch es cloud ID
                    es_connection_url = self.index_config.get("endpoint")
                    logger.info(f"Parsed elasticsearch endpoint: {es_connection_url}")

                    return ElasticsearchStore(
                        index_name=self.index_config.get("index"),
                        es_url=es_connection_url,
                        es_api_key=connection_credential.key,
                        embedding=self.get_langchain_embeddings(credential=credential),
                        vector_query_field=self.index_config.get("field_mapping", {}).get("embedding", "contentVector"),
                        query_field=self.index_config.get("field_mapping", {}).get("content", "text"),
                    )
                except Exception as e:
                    logger.warn(
                        f"Failed to get elasticsearch vectorstore due to {e}, try again by passing in `embedding` as an Embeddings object."
                    )
                    raise
            elif index_kind == IndexKinds.Qdrant:
                try:
                    from azure.core.credentials import AzureKeyCredential
                    from langchain_qdrant import Qdrant
                    from qdrant_client import QdrantClient

                    connection_credential = get_connection_credential(self.index_config, credential=credential)
                    if not isinstance(connection_credential, AzureKeyCredential):
                        raise ValueError(
                            f"Expected credential to qdrant index to be an AzureKeyCredential, instead got: {type(connection_credential)}"
                        )
                    # Get the connection credential to connect to qdrant host url
                    qdrant_host_url = self.index_config.get("endpoint")
                    qdrant_collection_name = self.index_config.get("index")

                    client = QdrantClient(url=qdrant_host_url, api_key=connection_credential.key)

                    return Qdrant(
                        client=client,
                        collection_name=qdrant_collection_name,
                        embeddings=self.get_langchain_embeddings(credential=credential),
                        content_payload_key=self.index_config.get("field_mapping", {}).get("content", "text"),
                        # None is treated as the default vector on qdrant side
                        vector_name=self.index_config.get("field_mapping", {}).get("embedding", None),
                    )
                except Exception as e:
                    logger.error(f"Failed to create Qdrant vectorstore due to: {e}")
                    raise
            elif index_kind == IndexKinds.Pinecone:
                try:
                    from azure.core.credentials import AzureKeyCredential
                    from langchain_pinecone import PineconeVectorStore
                    from pinecone import Pinecone

                    connection_credential = get_connection_credential(self.index_config, credential=credential)
                    if not isinstance(connection_credential, AzureKeyCredential):
                        raise ValueError(
                            f"Expected credential to Pinecone index to be an AzureKeyCredential, instead got: {type(connection_credential)}"
                        )

                    pc = Pinecone(api_key=connection_credential.key)
                    index_stats = pc.describe_index(self.index_config.get("index"))
                    logger.info(f"Pinecone index {self.index_config.get('index')} with stats {str(index_stats)}")
                    activity_logger.info("Pinecone index", extra={"properties": {"stats": str(index_stats)}})

                    return PineconeVectorStore(
                        embedding=self.get_langchain_embeddings(credential=credential),
                        text_key=self.index_config.get("field_mapping", {}).get("content", "text"),
                        pinecone_api_key=connection_credential.key,
                        index_name=self.index_config.get("index"),
                        namespace=self.index_config.get("namespace")
                        if self.index_config.get("namespace") != "default"
                        else None,
                    )
                except Exception as e:
                    logger.error(f"Failed to create Pinecone vectorstore due to: {e}")
                    raise
            elif index_kind == IndexKinds.Milvus:
                try:
                    from azure.core.credentials import AzureKeyCredential

                    try:
                        from langchain.vectorstores.milvus import Milvus
                    except ImportError:
                        from langchain_community.vectorstores.milvus import Milvus

                    connection_credential = get_connection_credential(self.index_config, credential=credential)
                    if not isinstance(connection_credential, AzureKeyCredential):
                        raise ValueError(
                            f"Invalid workspace connection credential type: {type(connection_credential)}. Expected: AzureKeyCredential"
                        )
                    return Milvus(
                        embedding_function=self.get_langchain_embeddings(credential=credential),
                        collection_name=self.index_config.get("index"),
                        connection_args={"uri": self.index_config.get("uri"), "token": connection_credential.key},
                        primary_field="id",
                        text_field=self.index_config.get("field_mapping", {}).get("content", "content"),
                        vector_field=self.index_config.get("field_mapping", {}).get("embedding", "contentVector"),
                    )
                except Exception as e:
                    logger.error(f"Failed to create Milvus vectorstore due to: {e}")
                    raise
            elif index_kind == IndexKinds.AzureCosmosDBforMongoDBvCore or index_kind == IndexKinds.MongoDB:
                try:
                    from azure.core.credentials import AzureKeyCredential

                    from azureml.rag.tasks.update_azure_cosmos_mongo_vcore import get_mongo_client

                    connection_credential = get_connection_credential(self.index_config, credential=credential)
                    if not isinstance(connection_credential, AzureKeyCredential):
                        raise ValueError(
                            f"Expected credential to mongo db index to be an AzureKeyCredential, instead got: {type(connection_credential)}"
                        )

                    mongo_client = get_mongo_client(connection_credential.key)
                    mongo_collection = mongo_client[self.index_config.get("database")][
                        self.index_config.get("collection")
                    ]

                    if index_kind == IndexKinds.AzureCosmosDBforMongoDBvCore:
                        try:
                            from langchain.vectorstores.azure_cosmos_db import AzureCosmosDBVectorSearch
                        except ImportError:
                            from langchain_community.vectorstores.azure_cosmos_db import AzureCosmosDBVectorSearch
                        return AzureCosmosDBVectorSearch(
                            mongo_collection,
                            self.get_langchain_embeddings(credential=credential),
                            index_name=self.index_config.get("index"),
                            text_key=self.index_config.get("field_mapping", {}).get("content", "content"),
                            embedding_key=self.index_config.get("field_mapping", {}).get("embedding", "contentVector"),
                        )
                    else:
                        if langchain_pkg_version > pkg_version.parse("0.0.25"):
                            from langchain.vectorstores import MongoDBAtlasVectorSearch
                        else:
                            from langchain_community.vectorstores.mongodb_atlas import MongoDBAtlasVectorSearch

                        return MongoDBAtlasVectorSearch(
                            mongo_collection,
                            self.get_langchain_embeddings(credential=credential),
                            index_name=self.index_config.get("search_index"),
                            text_key=self.index_config.get("field_mapping", {}).get("content", "content"),
                            embedding_key=self.index_config.get("field_mapping", {}).get("embedding", "contentVector"),
                        )

                except Exception as e:
                    logger.error(f"Failed to create Mongo DB vectorstore due to: {e}")
                    raise
            elif index_kind == IndexKinds.Weaviate:
                try:
                    import weaviate
                    from azure.core.credentials import AzureKeyCredential
                    from langchain_weaviate.vectorstores import WeaviateVectorStore

                    connection_credential = get_connection_credential(self.index_config, credential=credential)
                    if not isinstance(connection_credential, AzureKeyCredential):
                        raise ValueError(
                            f"Expected credential to Weaviate index to be an AzureKeyCredential, instead got: {type(connection_credential)}"
                        )

                    weaviate_client = weaviate.connect_to_wcs(
                        cluster_url=self.index_config.get("cluster_url"),
                        auth_credentials=weaviate.auth.AuthApiKey(connection_credential.key),
                    )

                    return WeaviateVectorStore(
                        client=weaviate_client,
                        index_name=self.index_config.get("collection"),
                        text_key=self.index_config.get("field_mapping", {}).get("content"),
                        embedding=self.get_langchain_embeddings(credential=credential),
                    )
                except Exception as e:
                    logger.error(f"Failed to create Weaviate vectorstore due to: {e}")
                    raise
            elif index_kind == IndexKinds.AzureCosmosDBforNoSQL:
                try:
                    from azure.core.credentials import AzureKeyCredential

                    from azureml.rag.langchain.vendor.vectorstores.cosmosdbnosql import AzureCosmosDBNoSQLVectorSearch
                    from azureml.rag.tasks.update_azure_cosmos_nosql import get_cosmosdb_client

                    connection_credential = get_connection_credential(self.index_config, credential=credential)
                    if not isinstance(connection_credential, AzureKeyCredential):
                        raise ValueError(
                            f"Expected credential to cosmosdb nosql index to be an AzureKeyCredential, instead got: {type(connection_credential)}"
                        )

                    cosmos_client = get_cosmosdb_client(connection_string=connection_credential.key)

                    return AzureCosmosDBNoSQLVectorSearch(
                        cosmos_client=cosmos_client,
                        database_name=self.index_config.get("database"),
                        container_name=self.index_config.get("container"),
                        text_key=self.index_config.get("field_mapping", {}).get("content"),
                        embedding=self.get_langchain_embeddings(credential=credential),
                        embedding_key=self.index_config.get("field_mapping", {}).get("embedding"),
                    )
                except Exception as e:
                    logger.error(f"Failed to create cosmosdb nosql vectorstore due to: {e}")
                    raise
            else:
                raise ValueError(f"Unknown index kind: {index_kind}")

    def as_langchain_retriever(self, credential: Optional[TokenCredential] = None, **kwargs):
        """Converts MLIndex to a retriever object that can be used with langchain, may download files."""
        index_kind = self.index_config.get("kind", None)
        if index_kind == IndexKinds.AzureCognitiveSearch:
            if self.index_config.get("field_mapping", {}).get("embedding", None) is None:
                from azureml.rag.langchain.acs import AzureCognitiveSearchVectorStore

                connection_credential = get_connection_credential(self.index_config, credential=credential)

                endpoint = self.index_config.get("endpoint", None)
                if not endpoint:
                    endpoint = get_target_from_connection(
                        get_connection_by_id_v2(self.index_config["connection"]["id"], credential=credential)
                    )
                return AzureCognitiveSearchVectorStore(
                    index_name=self.index_config.get("index"),
                    endpoint=endpoint,
                    embeddings=self.get_langchain_embeddings(),
                    field_mapping=self.index_config.get("field_mapping", {}),
                    credential=connection_credential,
                ).as_retriever(**kwargs)

            # Using AzureSearchProxy to replace the vectorstore in AzureSearchVectorStoreRetriever
            from langchain_community.vectorstores.azuresearch import AzureSearchVectorStoreRetriever

            from azureml.rag.utils.acs import AzureSearchVectorStoreRetrieverProxy

            azuresearch_proxy = self.as_langchain_vectorstore(credential=credential)
            azuresearch = azuresearch_proxy.get_azuresearch_instance()

            search_type = kwargs.get("search_type", azuresearch.search_type)
            kwargs["search_type"] = search_type
            tags = kwargs.pop("tags", None) or []
            tags.extend(azuresearch._get_retriever_tags())
            retriever = AzureSearchVectorStoreRetriever(vectorstore=azuresearch, **kwargs, tags=tags)
            module_settings = azuresearch_proxy.get_module_settings()
            return AzureSearchVectorStoreRetrieverProxy(retriever_instance=retriever, module_settings=module_settings)
        elif index_kind in [
            IndexKinds.FAISS,
            IndexKinds.Pinecone,
            IndexKinds.Milvus,
            IndexKinds.MongoDB,
            IndexKinds.AzureCosmosDBforMongoDBvCore,
            IndexKinds.AzureCosmosDBforNoSQL,
        ]:
            return self.as_langchain_vectorstore(credential=credential).as_retriever(**kwargs)
        else:
            raise ValueError(f"Unknown index kind: {index_kind}")

    def as_native_index_client(self, credential: Optional[TokenCredential] = None):
        """
        Converts MLIndex config into a client for the underlying Index, may download files.

        # An azure.search.documents.SearchClient for acs indexes
        # or an azureml.rag.indexes.indexFaissAndDocStore for faiss indexes.
        """
        index_kind = self.index_config.get("kind", None)
        if index_kind == IndexKinds.AzureCognitiveSearch:
            connection_credential = get_connection_credential(self.index_config, credential=credential)

            from azure.search.documents import SearchClient

            return SearchClient(
                endpoint=self.index_config.get("endpoint"),
                index_name=self.index_config.get("index"),
                credential=connection_credential,
                user_agent=f"azureml-rag=={version}/mlindex",
                api_version=self.index_config.get("api_version", ACS_API_VERSION),
            )
        elif index_kind == IndexKinds.FAISS:
            from azureml.rag.indexes.faiss import FaissAndDocStore

            embeddings = self.get_langchain_embeddings(credential=credential)

            return FaissAndDocStore.load(self.index_config.get("path", self.base_uri), embeddings.embed_query)
        elif index_kind == IndexKinds.Pinecone:
            from azure.core.credentials import AzureKeyCredential
            from pinecone import Pinecone

            connection_credential = get_connection_credential(self.index_config, credential=credential)
            if not isinstance(connection_credential, AzureKeyCredential):
                raise ValueError(
                    f"Expected credential to Pinecone index to be an AzureKeyCredential, instead got: {type(connection_credential)}"
                )

            pc = Pinecone(api_key=connection_credential.key)
            return pc.Index(self.index_config.get("index"))
        elif index_kind == IndexKinds.Milvus:
            from azure.core.credentials import AzureKeyCredential
            from pymilvus import MilvusClient

            from azureml.rag.tasks.update_milvus import MILVUS_URI_KEY

            connection_credential = get_connection_credential(self.index_config, credential=credential)
            if not isinstance(connection_credential, AzureKeyCredential):
                raise ValueError(
                    f"Invalid workspace connection credential type: {type(connection_credential)}. Expected: AzureKeyCredential"
                )
            return MilvusClient(uri=self.index_config.get(MILVUS_URI_KEY), token=connection_credential.key)
        elif index_kind == IndexKinds.AzureCosmosDBforMongoDBvCore or index_kind == IndexKinds.MongoDB:
            from azure.core.credentials import AzureKeyCredential

            from azureml.rag.tasks.update_azure_cosmos_mongo_vcore import get_mongo_client

            connection_credential = get_connection_credential(self.index_config, credential=credential)
            if not isinstance(connection_credential, AzureKeyCredential):
                raise ValueError(
                    f"Expected credential to Azure Cosmos Mongo vCore index to be an AzureKeyCredential, instead got: {type(connection_credential)}"
                )

            mongo_client = get_mongo_client(connection_credential.key)
            return mongo_client[self.index_config.get("database")][self.index_config.get("collection")]
        elif index_kind == IndexKinds.AzureCosmosDBforNoSQL:
            from azure.core.credentials import AzureKeyCredential

            from azureml.rag.tasks.update_azure_cosmos_nosql import get_cosmosdb_client

            connection_credential = get_connection_credential(self.index_config, credential=credential)
            if not isinstance(connection_credential, AzureKeyCredential):
                raise ValueError(
                    f"Expected credential to Azure Cosmos for NoSql index to be an AzureKeyCredential, instead got: {type(connection_credential)}"
                )

            cosmos_client = get_cosmosdb_client(connection_credential.key)
            database_client = cosmos_client.get_database_client(self.index_config.get("database"))
            return database_client.get_container_client(self.index_config.get("container"))
        else:
            raise ValueError(f"Unknown index kind: {index_kind}")

    def __repr__(self):
        """Returns a string representation of the MLIndex object."""
        return yaml.dump(
            {
                "index": self.index_config,
                "embeddings": self.embeddings_config,
            }
        )

    def override_connections(
        self,
        embedding_connection: Optional[Union[str, Connection]] = None,
        index_connection: Optional[Union[str, Connection]] = None,
        credential: Optional[TokenCredential] = None,
    ) -> "MLIndex":
        """
        Override the connections used by the MLIndex.

        Args:
        ----
            embedding_connection: Optional connection to use for embeddings model
            index_connection: Optional connection to use for index
            credential: Optional credential to use when resolving connection information

        """
        if embedding_connection:
            if self.embeddings_config.get("key") is not None:
                self.embeddings_config.pop("key")

            if embedding_connection.__class__.__name__ == "AzureOpenAIConnection":
                # PromptFlow Connection
                self.embeddings_config["connection_type"] = "inline"
                self.embeddings_config["connection"] = {
                    "key": embedding_connection.secrets.get("api_key"),
                    "api_base": embedding_connection.api_base,
                    "api_type": embedding_connection.api_type,
                }
            else:
                self.embeddings_config["connection_type"] = "workspace_connection"
                if isinstance(embedding_connection, str):
                    from azureml.rag.utils.connections import get_connection_by_id_v2

                    embedding_connection = get_connection_by_id_v2(embedding_connection, credential=credential)
                self.embeddings_config["connection"] = {"id": get_id_from_connection(embedding_connection)}
        if index_connection:
            if self.index_config["kind"] != IndexKinds.AzureCognitiveSearch:
                print("Index kind is not acs, ignoring override for connection")
            else:
                self.index_config["connection_type"] = "workspace_connection"
                if isinstance(index_connection, str):
                    from azureml.rag.utils.connections import get_connection_by_id_v2

                    index_connection = get_connection_by_id_v2(index_connection, credential=credential)
                self.index_config["connection"] = {"id": get_id_from_connection(index_connection)}
        self.save(just_config=True)
        return self

    def set_embeddings_connection(
        self,
        connection: Optional[Union[str, Connection]],
        credential: Optional[TokenCredential] = None,
    ) -> "MLIndex":
        """Set the embeddings connection used by the MLIndex."""
        return self.override_connections(embedding_connection=connection)

    @staticmethod
    def from_files(
        source_uri: str,
        source_glob: str = "**/*",
        chunk_size: int = 1000,
        chunk_overlap: int = 0,
        citation_url: Optional[str] = None,
        citation_replacement_regex: Optional[Dict[str, str]] = None,
        embeddings_model: str = "hugging_face://model/sentence-transformers/all-mpnet-base-v2",
        embeddings_connection: Optional[str] = None,
        embeddings_container: Optional[Union[str, Path]] = None,
        index_type: str = "faiss",
        index_connection: Optional[str] = None,
        index_config: Dict[str, Any] = {},
        output_path: Optional[Union[str, Path]] = None,
        credential: Optional[TokenCredential] = None,
    ) -> "MLIndex":
        r"""
        Create a new MLIndex from a repo.

        Args:
        ----
            source_uri: Iterator of documents to index
            source_glob: Glob pattern to match files to index
            chunk_size: Size of chunks to split documents into
            chunk_overlap: Size of overlap between chunks
            citation_url: Optional url to replace citation urls with
            citation_replacement_regex: Optional regex to use to replace citation urls,
              e.g. `{"match_pattern": "(.*)/articles/(.*)(\.[^.]+)$", "replacement_pattern": "\1/\2"}`
            embeddings_model: Name of embeddings model to use, expected format
              `azure_open_ai://deployment/.../model/text-embedding-ada-002` or `hugging_face://model/all-mpnet-base-v2`
            embeddings_connection: Optional connection to use for embeddings model
            embeddings_container: Optional path to location where un-indexed embeddings can be saved/loaded.
            index_type: Type of index to use, e.g. faiss
            index_connection: Optional connection to use for index
            index_config: Config for index, e.g. index_name or field_mapping for acs

        Returns:
        -------
            MLIndex

        """
        from azureml.rag.documents import DocumentChunksIterator, split_documents

        with track_activity(logger, "MLIndex.from_files"):
            chunked_documents = DocumentChunksIterator(
                files_source=source_uri,
                glob=source_glob,
                base_url=citation_url,
                document_path_replacement_regex=citation_replacement_regex,
                chunked_document_processors=[
                    lambda docs: split_documents(
                        docs,
                        splitter_args={"chunk_size": chunk_size, "chunk_overlap": chunk_overlap, "use_rcts": False},
                    )
                ],
            )

            mlindex = MLIndex.from_documents(
                chunked_documents,
                embeddings_model=embeddings_model,
                embeddings_connection=embeddings_connection,
                embeddings_container=embeddings_container,
                index_type=index_type,
                index_connection=index_connection,
                index_config=index_config,
                output_path=output_path,
                credential=credential,
            )

        return mlindex

    @staticmethod
    def from_documents(
        documents: Union[Iterator[Document], BaseLoader, DocumentChunksIterator],
        embeddings_model: str = "hugging_face://model/sentence-transformers/all-mpnet-base-v2",
        embeddings_connection: Optional[str] = None,
        embeddings_container: Optional[Union[str, Path]] = None,
        index_type: str = "faiss",
        index_connection: Optional[str] = None,
        index_config: Dict[str, Any] = {},
        output_path: Optional[Union[str, Path]] = None,
        credential: Optional[TokenCredential] = None,
    ) -> "MLIndex":
        """
        Create a new MLIndex from documents.

        Args:
        ----
            documents: Iterator of documents to index
            index_kind: Kind of index to use
            embeddings_model: Name of embeddings model to use, expected format
              `azure_open_ai://deployment/.../model/text-embedding-ada-002` or `hugging_face://model/all-mpnet-base-v2`
            embeddings_container: Optional path to location where un-indexed embeddings can be saved/loaded.
            index_type: Type of index to use, e.g. faiss
            index_connection: Optional connection to use for index
            index_config: Config for index, e.g. index_name or field_mapping for acs
            output_path: Optional path to save index to

        Returns:
        -------
            MLIndex

        """
        import time

        embeddings = None
        # TODO: Move this logic to load from embeddings_container into EmbeddingsContainer
        try:
            if embeddings_container is not None:
                if isinstance(embeddings_container, str) and "://" in embeddings_container:
                    from fsspec.core import url_to_fs

                    fs, uri = url_to_fs(embeddings_container)
                else:
                    embeddings_container = Path(embeddings_container)
                    previous_embeddings_dir_name = None
                    try:
                        previous_embeddings_dir_name = str(
                            max(
                                [dir for dir in embeddings_container.glob("*") if dir.is_dir()], key=os.path.getmtime
                            ).name
                        )
                    except Exception as e:
                        logger.warning(
                            f"failed to get latest folder from {embeddings_container} with {e}.", extra={"print": True}
                        )
                        pass
                    if previous_embeddings_dir_name is not None:
                        try:
                            embeddings = EmbeddingsContainer.load(previous_embeddings_dir_name, embeddings_container)
                        except Exception as e:
                            logger.warning(
                                f"failed to load embeddings from {embeddings_container} with {e}.",
                                extra={"print": True},
                            )
                            pass
        finally:
            if embeddings is None:
                logger.info("Creating new EmbeddingsContainer")
                if isinstance(embeddings_model, str):
                    connection_args = {}
                    if "open_ai" in embeddings_model:
                        from azureml.rag.utils.connections import get_connection_by_id_v2

                        if embeddings_connection:
                            if isinstance(embeddings_connection, str):
                                embeddings_connection = get_connection_by_id_v2(
                                    embeddings_connection, credential=credential
                                )
                            connection_args = {
                                "connection_type": "workspace_connection",
                                "connection": {"id": get_id_from_connection(embeddings_connection)},
                                "endpoint": embeddings_connection.target
                                if hasattr(embeddings_connection, "target")
                                else embeddings_connection["properties"]["target"],
                            }
                        else:
                            connection_args = {
                                "connection_type": "environment",
                                "connection": {"key": "OPENAI_API_KEY"},
                                "endpoint": os.getenv("OPENAI_API_BASE"),
                            }
                            if os.getenv("OPENAI_API_TYPE"):
                                connection_args["api_type"] = os.getenv("OPENAI_API_TYPE")
                            if os.getenv("OPENAI_API_VERSION"):
                                connection_args["api_version"] = os.getenv("OPENAI_API_VERSION")
                            if os.getenv("AZURE_OPENAI_API_VERSION"):
                                connection_args["api_version"] = os.getenv("AZURE_OPENAI_API_VERSION")

                    embeddings = EmbeddingsContainer.from_uri(
                        embeddings_model, credential=credential, **connection_args
                    )
                else:
                    raise ValueError(f"Unknown embeddings model: {embeddings_model}")
                    # try:
                    #     import sentence_transformers
                    #     if isinstance(embeddings_model, sentence_transformers.SentenceTransformer):
                    #         embeddings = EmbeddingsContainer.from_sentence_transformer(embeddings_model)
                    # except Exception as e:
                    #     logger.warning(f"Failed to load sentence_transformers with {e}.")

        pre_embed = time.time()
        embeddings = embeddings.embed(documents)
        post_embed = time.time()
        logger.info(f"Embedding took {post_embed - pre_embed} seconds")

        if embeddings_container is not None:
            now = datetime.now()
            # TODO: This means new snapshots will be created for every run,
            #       ideally there'd be a use container as readonly vs persist snapshot option
            embeddings.save(
                str(
                    embeddings_container
                    / f"{now.strftime('%Y%m%d')}_{now.strftime('%H%M%S')}_{str(uuid.uuid4()).split('-')[0]}"
                )
            )

        mlindex = MLIndex.from_embeddings_container(
            embeddings,
            index_type=index_type,
            index_connection=index_connection,
            index_config=index_config,
            output_path=output_path,
            credential=credential,
        )

        return mlindex

    @staticmethod
    def from_embeddings_container(
        embeddings: EmbeddingsContainer,
        index_type: str,
        index_connection: Optional[str] = None,
        index_config: Dict[str, Any] = {},
        output_path: Optional[Union[str, Path]] = None,
        credential: Optional[TokenCredential] = None,
    ) -> "MLIndex":
        """
        Create a new MLIndex from embeddings.

        Args
        ----
            embeddings: EmbeddingsContainer to index
            index_type: Type of index to use, e.g. faiss
            index_connection: Optional connection to use for index
            index_config: Config for index, e.g. index_name or field_mapping for acs
            output_path: Optional path to save index to
            credential: Optional credential to use when resolving connection information

        Returns
        -------
            MLIndex

        """
        if output_path is None:
            output_path = Path.cwd() / f"{index_type}_{embeddings.kind}_index"
        if index_type == IndexKinds.FAISS:
            embeddings.write_as_faiss_mlindex(
                output_path=output_path, engine=index_config.get("engine", "indexes.faiss.FaissAndDocStore")
            )

            mlindex = MLIndex(
                uri=Path(output_path),
            )
        elif index_type == IndexKinds.AzureCognitiveSearch:
            from azureml.rag.tasks.update_acs import create_index_from_raw_embeddings
            from azureml.rag.utils.connections import get_connection_by_id_v2

            if not index_connection:
                index_config = {
                    **index_config,
                    **{
                        "endpoint": os.getenv("AZURE_COGNITIVE_SEARCH_TARGET"),
                        "api_version": ACS_API_VERSION,
                    },
                }
                connection_args = {
                    "connection_type": "environment",
                    "connection": {"key": "AZURE_COGNITIVE_SEARCH_KEY"},
                }
            else:
                if isinstance(index_connection, str):
                    index_connection = get_connection_by_id_v2(index_connection, credential=credential)
                index_config = {
                    **index_config,
                    **{
                        "endpoint": get_target_from_connection(index_connection),
                        "api_version": get_metadata_from_connection(index_connection).get(
                            "apiVersion", ACS_API_VERSION
                        ),
                    },
                }
                connection_args = {
                    "connection_type": "workspace_connection",
                    "connection": {"id": get_id_from_connection(index_connection)},
                }

            mlindex = create_index_from_raw_embeddings(
                embeddings,
                index_config,
                connection=connection_args,
                output_path=str(output_path),
                credential=credential,
            )
        elif index_type == IndexKinds.Pinecone:
            from azureml.rag.tasks.update_pinecone import create_index_from_raw_embeddings

            if not index_connection:
                index_config["environment"] = os.getenv("PINECONE_ENVIRONMENT")
                connection_args = {"connection_type": "environment", "connection": {"key": "PINECONE_API_KEY"}}
            else:
                if isinstance(index_connection, str):
                    index_connection = get_connection_by_id_v2(index_connection, credential=credential)
                index_config["environment"] = get_metadata_from_connection(index_connection).get("environment")
                connection_args = {
                    "connection_type": "workspace_connection",
                    "connection": {"id": get_id_from_connection(index_connection)},
                }
            mlindex = create_index_from_raw_embeddings(
                embeddings,
                index_config,
                connection=connection_args,
                output_path=str(output_path),
                credential=credential,
            )
        elif index_type == IndexKinds.Milvus:
            from azureml.rag.tasks.update_milvus import (
                MILVUS_COLLECTION_NAME_KEY,
                MILVUS_URI_KEY,
                create_index_from_raw_embeddings,
                try_override_milvus_config_with_connection_metadata,
            )

            if not index_connection:
                if os.get("MILVUS_URI") is not None:
                    index_config[MILVUS_URI_KEY] = os.get("MILVUS_URI")
                if os.get("MILVUS_COLLECTION_NAME") is not None:
                    index_config[MILVUS_COLLECTION_NAME_KEY] = os.get("MILVUS_COLLECTION_NAME")
                connection_args = {"connection_type": "environment", "connection": {"key": "MILVUS_API_KEY"}}
            else:
                if isinstance(index_connection, str):
                    index_connection = get_connection_by_id_v2(index_connection, credential=credential)
                connection_metadata = get_metadata_from_connection(index_connection)
                try_override_milvus_config_with_connection_metadata(index_config, MILVUS_URI_KEY, connection_metadata)
                try_override_milvus_config_with_connection_metadata(
                    index_config, MILVUS_COLLECTION_NAME_KEY, connection_metadata
                )
                connection_args = {
                    "connection_type": "workspace_connection",
                    "connection": {"id": get_id_from_connection(index_connection)},
                }
            mlindex = create_index_from_raw_embeddings(
                embeddings,
                index_config,
                connection=connection_args,
                output_path=str(output_path),
                credential=credential,
            )
        elif index_type == IndexKinds.AzureCosmosDBforMongoDBvCore:
            from azureml.rag.tasks.update_azure_cosmos_mongo_vcore import create_index_from_raw_embeddings

            if not index_connection:
                connection_args = {
                    "connection_type": "environment",
                    "connection": {"key": "AZURE_COSMOS_MONGO_VCORE_CONNECTION_STRING"},
                }
            else:
                if isinstance(index_connection, str):
                    index_connection = get_connection_by_id_v2(index_connection, credential=credential)
                connection_args = {
                    "connection_type": "workspace_connection",
                    "connection": {"id": get_id_from_connection(index_connection)},
                }
            mlindex = create_index_from_raw_embeddings(
                embeddings,
                index_config,
                connection=connection_args,
                output_path=str(output_path),
                credential=credential,
            )
        else:
            raise ValueError(f"Unknown index type: {index_type}")

        return mlindex

    def save(self, output_uri: Optional[str], just_config: bool = False):
        """
        Save the MLIndex to a uri.

        Will use uri MLIndex was loaded from if `output_uri` not set.
        """
        # Use fsspec to create MLIndex yaml file at output_uri and call save on _underlying_index if present
        try:
            import fsspec

            mlindex_file = fsspec.open(f"{output_uri.rstrip('/')}/MLIndex", "w")
            # parse yaml to dict
            with mlindex_file as f:
                yaml.safe_dump({"embeddings": self.embeddings_config, "index": self.index_config}, f)

            if not just_config:
                files = fsspec.open_files(f"{self.base_uri}/*")
                files += fsspec.open_files(f"{self.base_uri}/**/*")
                for file in files:
                    if file.path.endswith("MLIndex"):
                        continue

                    with file.open() as src, fsspec.open(
                        f"{output_uri.rstrip('/')}/{file.path.replace(self.base_uri, '').lstrip('/')}", "wb"
                    ) as dest:
                        dest.write(src.read())
        except Exception as e:
            raise ValueError(f"Could not save MLIndex: {e}") from e
