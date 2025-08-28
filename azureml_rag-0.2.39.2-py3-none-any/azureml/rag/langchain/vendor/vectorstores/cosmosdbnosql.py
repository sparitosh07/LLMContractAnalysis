from azure.cosmos import CosmosClient
from enum import Enum
from typing import (
    Any,
    Iterable,
    List,
    Optional,
    Tuple
)
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore


class CosmosDBNoSQLSimilarityType(str, Enum):
    """Cosmos DB Similarity Type as enumerator."""

    COSINE = "cosine"
    """Cosine similarity"""
    DOT_PRODUCT = "dotproduct"
    """Dot product similarity"""
    EUCLIDEAN = "euclidean"
    """Euclidean distance similarity"""


class CosmosDBNoSQLIndexType(str, Enum):
    """Cosmos DB for NoSQL vector index type as enumerator."""

    FLAT = "flat"
    """Stores vectors on the same index as other indexed properties."""
    QUANTIZED_FLAT = "quantizedFlat"
    """Quantizes (compresses) vectors before storing on the index."""
    DISK_ANN = "diskANN"
    """Creates an index based on DiskANN for fast and efficient approximate search."""


class AzureCosmosDBNoSQLVectorSearch(VectorStore):
    """`Azure Cosmos DB for NoSQL` vector store.


    To use, you should have:
        - the ``azure-cosmos`` python package installed

    You can read more about vector search using AzureCosmosDBNoSQL here:
    https://learn.microsoft.com/en-us/azure/cosmos-db/nosql/vector-search
    """

    def __init__(
        self,
        cosmos_client: CosmosClient,
        database_name: str,
        container_name: str,
        embedding: Embeddings,
        *,
        text_key: str = "textContent",
        embedding_key: str = "vectorContent",
    ):
        """Constructor for AzureCosmosDBVectorSearch

        Args:
            cosmos_client: Azure CosmosDB client
            database_name: CosmosDB database name to add the texts to.
            container_name: CosmosDB container name to add the texts to.
            embedding: Text embedding model to use.
            index_name: Name of the Atlas Search index.
            text_key: MongoDB field that will contain the text
                for each document.
            embedding_key: MongoDB field that will contain the embedding
                for each document.
        """
        self._database_name = database_name
        self._container_name = container_name
        self._database = cosmos_client.get_database_client(database_name)
        self._container = self._database.get_container_client(container_name)
        self._embedding = embedding
        self._text_key = text_key
        self._embedding_key = embedding_key

    @property
    def embeddings(self) -> Embeddings:
        return self._embedding

    def get_index_name(self) -> str:
        """Returns the index name

        Returns:
            Returns the index name

        """
        return self._container_name
    
    @classmethod
    def from_connection_string(
        cls,
        connection_string: str,
        database_name: str,
        container_name: str,
        embedding: Embeddings,
        **kwargs: Any,
    ) -> "AzureCosmosDBNoSQLVectorSearch":
        """Creates an Instance of AzureCosmosDBNoSQLVectorSearch from a Connection String

        Args:
            connection_string: The MongoDB vCore instance connection string
            namespace: The namespace (database.collection)
            embedding: The embedding utility
            **kwargs: Dynamic keyword arguments

        Returns:
            an instance of the vector store

        """
        client: CosmosClient = CosmosClient.from_connection_string(conn_str=connection_string)
        return cls(client, database_name, container_name, embedding)
    
    def index_exists(self) -> bool:
        """Verifies if the specified index name during instance
            construction exists on the collection

        Returns:
          Returns True on success and False if no such index exists
            on the collection
        """
        containers = self._database.list_containers()
        index_name = self.get_index_name()

        for container in containers:
            current_index_name = container.get('id')
            if current_index_name == index_name:
                return True

        return False

    def _similarity_search_with_score(
        self,
        embeddings: List[float],
        k: int = 4,
        search_type: CosmosDBNoSQLSimilarityType = CosmosDBNoSQLSimilarityType.COSINE,
        score_threshold: float = 0.0,
    ) -> List[Tuple[Document, float]]:
        """Returns a list of documents with their scores

        Args:
            embeddings: The query vector
            k: the number of documents to return
            search_type: The vector search method that will be used in similarity search.
            Defaults to cosine.
            score_threshold: (Optional[float], optional): Minimum similarity
            between selected documents and the query vector. Defaults to 0

        Returns:
            List of documents with score similar to the query text with
            similarity score in float. Sorts in order of most-similar
            to least-similar.
        """
        vectorsearch_method = search_type
        query = self._build_sql_query()

        results = self._container.query_items(
            query=query,
            parameters=[
                {"name": "@embedding", "value": embeddings},
                {"name": "@num_results", "value": k},
                {"name": "@distance_function", "value": vectorsearch_method}
            ],
            enable_cross_partition_query=True
        )

        docs = []
        for row in results:
            row_data = row["Document"]
            score = float(row["SimilarityScore"])
            if score < score_threshold:
                continue
            docs.append(
                (
                    Document(
                        page_content=self.__fetch_property_from_path(row_data, self._text_key),
                        metadata=row_data
                    ),
                    score
                )
            )
        return docs
        
    def _build_sql_query(self) -> str:
        embedding_keys = self._list_embedding_keys()
        if self._embedding_key not in embedding_keys:
            raise ValueError(
                f"Failed to build query: embedding field: {self._embedding_key}"
                " is not in container vector embedding policy"
            )
        
        processed_key = ".".join(self._embedding_key.strip("/").split("/"))
        query = f"""
        SELECT TOP @num_results VALUE {{
            "Document": c,
            "SimilarityScore": VectorDistance(c.{processed_key}, @embedding, false, {{'distanceFunction': @distance_function}})
        }}
        FROM c
        ORDER BY VectorDistance(c.{processed_key}, @embedding, false, {{'distanceFunction': @distance_function}})
        """  # noqa: E501
        return query
    
    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        search_type: CosmosDBNoSQLSimilarityType = CosmosDBNoSQLSimilarityType.COSINE,
        score_threshold: float = 0.0,
    ) -> List[Tuple[Document, float]]:
        """Return docs most similar to query.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            search_type: The vector search method that will be used in similarity search.
            Defaults to cosine.
            score_threshold: (Optional[float], optional): Minimum similarity
            between selected documents and the query vector. Defaults to 0

        Returns:
            List of documents with score most similar to the query text with
            similarity score in float. Sorts in order of most-similar
            to least-similar.
        """    
        embeddings = self._embedding.embed_query(query)
        docs = self._similarity_search_with_score(
            embeddings=embeddings,
            k=k,
            search_type=search_type,
            score_threshold=score_threshold,
        )
        return docs

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        search_type: CosmosDBNoSQLSimilarityType = CosmosDBNoSQLSimilarityType.COSINE,
        score_threshold: float = 0.0,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs most similar to query.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            search_type: The vector search method that will be used in similarity search
            score_threshold: (Optional[float], optional): Minimum similarity
            between selected documents and the query vector. Defaults to 0

        Returns:
            List of documents most similar to the query text with
            similarity score in float. Sorts in order of most-similar
            to least-similar.
        """ 
        docs_and_scores = self.similarity_search_with_score(
            query,
            k=k,
            search_type=search_type,
            score_threshold=score_threshold,
        )
        return [doc for doc, _ in docs_and_scores]

    def _list_embedding_keys(self):
        embedding_keys = []
        try:
            properties = self._container.read()
            vector_embeddings = properties['vectorEmbeddingPolicy']['vectorEmbeddings']
            for vector_embedding in vector_embeddings:
                embedding_keys.append(vector_embedding['path'])
        except Exception as e:
            raise ValueError(f"Failed to list embedding keys, please check container policies, {e}")
        
        return embedding_keys

    def __fetch_property_from_path(self, doc: dict, path: str):
        try:
            keys = path.strip("/ ").split("/")
            res = doc
            for key in keys:
                res = res[key]
            return res
        except (KeyError, TypeError):
            raise ValueError(f"Can't find path: {path} in data")
    
    def add_texts(
            self,
            texts: Iterable[str],
            metadatas: Optional[List[dict]] = None,
            **kwargs: Any,
        ):
        """Run more texts through the embeddings and add to the vectorstore.

        Args:
            texts: Iterable of strings to add to the vectorstore.
            metadatas: Optional list of metadatas associated with the texts.

        Returns:
            List of ids from adding the texts into the vectorstore.
        """
        raise NotImplementedError()

    def from_texts(self):
        """Used to Load Documents into the collection

        Args:
            texts: The list of documents strings to load
            metadatas: The list of metadata objects associated with each document

        Returns:
            List of ids from adding the texts into the vectorstore.
        """
        raise NotImplementedError()
