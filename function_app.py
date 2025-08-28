import logging
import json
import os
import uuid
from typing import List, Dict, Any, Optional
import azure.functions as func
from openai import AzureOpenAI
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential

from config import get_app_config, AppConfig
from document_processing import DocumentProcessor
from utils.logging_utils import (
    track_activity, monitor_performance, 
    LoggingConfig, exception_handler, log_processing_stats
)


# Setup logging
LoggingConfig.setup_logging(level=logging.INFO)
logger = LoggingConfig.get_function_logger("process_document")

app = func.FunctionApp(http_auth_level=func.AuthLevel.FUNCTION)

# Global config (loaded once)
APP_CONFIG: Optional[AppConfig] = None


def get_app_config_cached() -> AppConfig:
    """Get cached app configuration."""
    global APP_CONFIG
    if APP_CONFIG is None:
        APP_CONFIG = get_app_config()
    return APP_CONFIG


@monitor_performance(logger, "embedding_generation")
def generate_embeddings_batch(
    chunks: List[Dict[str, Any]], 
    openai_client: AzureOpenAI, 
    config: AppConfig
) -> List[Dict[str, Any]]:
    """
    Generate embeddings for text chunks using Azure OpenAI with batching and retry logic.
    """
    embedded_chunks = []
    batch_size = min(config.processing.batch_size, 100)  # OpenAI limit
    
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        
        for retry in range(config.processing.max_retries):
            try:
                # Generate embeddings for batch
                texts = [chunk["content"] for chunk in batch]
                
                response = openai_client.embeddings.create(
                    model=config.openai.embedding_model,
                    input=texts
                )
                
                # Add embeddings to chunks
                for j, embedding_data in enumerate(response.data):
                    batch[j]["embedding"] = embedding_data.embedding
                    embedded_chunks.append(batch[j])
                
                logger.info(f"Generated embeddings for batch {i//batch_size + 1} ({len(batch)} chunks)")
                break
                
            except Exception as e:
                logger.warning(f"Embedding batch {i//batch_size + 1} failed (attempt {retry + 1}): {str(e)}")
                if retry == config.processing.max_retries - 1:
                    logger.error(f"Failed to generate embeddings for batch {i//batch_size + 1} after {config.processing.max_retries} attempts")
                    # Continue with next batch instead of failing completely
                    break
    
    logger.info(f"Successfully generated embeddings for {len(embedded_chunks)}/{len(chunks)} chunks")
    return embedded_chunks


@monitor_performance(logger, "search_index_update")
def update_search_index_batch(
    chunks: List[Dict[str, Any]], 
    search_client: SearchClient, 
    document_id: str,
    config: AppConfig
) -> Dict[str, Any]:
    """
    Update Azure Cognitive Search index with embedded chunks using batch operations.
    """
    documents = []
    
    for chunk in chunks:
        document = {
            "id": chunk["id"],
            "document_id": document_id,
            "content": chunk["content"],
            "embedding": chunk.get("embedding", []),
            "chunk_index": chunk["chunk_index"],
            "metadata": json.dumps(chunk.get("metadata", {})),
            "tokens": chunk.get("token_count", 0),
            "file_extension": chunk.get("file_extension", ""),
            "title": chunk.get("title", "")
        }
        documents.append(document)
    
    # Batch upload with retry logic
    batch_size = min(config.processing.batch_size, 1000)  # ACS limit
    total_uploaded = 0
    failed_uploads = 0
    
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        
        for retry in range(config.processing.max_retries):
            try:
                result = search_client.upload_documents(batch)
                
                # Count successful uploads
                successful = sum(1 for r in result if r.succeeded)
                total_uploaded += successful
                failed_uploads += len(batch) - successful
                
                logger.info(f"Uploaded batch {i//batch_size + 1}: {successful}/{len(batch)} documents succeeded")
                break
                
            except Exception as e:
                logger.warning(f"Search index batch {i//batch_size + 1} failed (attempt {retry + 1}): {str(e)}")
                if retry == config.processing.max_retries - 1:
                    logger.error(f"Failed to upload batch {i//batch_size + 1} after {config.processing.max_retries} attempts")
                    failed_uploads += len(batch)
    
    success = failed_uploads == 0
    return {
        "success": success,
        "uploaded_count": total_uploaded,
        "failed_count": failed_uploads,
        "total_documents": len(documents)
    }


@app.route(route="process_document", methods=["POST"])
@exception_handler(logger)
def process_document(req: func.HttpRequest) -> func.HttpResponse:
    """
    Azure Function to process text file: sophisticated chunking, embedding, and indexing.
    
    Expected request body:
    {
        "text": "your text content here",
        "filename": "document.txt",  
        "document_id": "optional_document_id",
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "content_type": "text/plain"
    }
    """
    with track_activity(logger, "process_document") as activity_logger:
        try:
            # Get configuration
            config = get_app_config_cached()
            
            # Parse and validate request
            req_body = req.get_json()
            if not req_body:
                return func.HttpResponse(
                    json.dumps({"error": "Request body is required"}),
                    status_code=400,
                    mimetype="application/json"
                )
            
            text_content = req_body.get('text', '').strip()
            if not text_content:
                return func.HttpResponse(
                    json.dumps({"error": "Text content is required and cannot be empty"}),
                    status_code=400,
                    mimetype="application/json"
                )
            
            # Extract parameters with defaults
            filename = req_body.get('filename', 'document.txt')
            document_id = req_body.get('document_id', str(uuid.uuid4()))
            chunk_size = req_body.get('chunk_size', config.processing.chunk_size)
            chunk_overlap = req_body.get('chunk_overlap', config.processing.chunk_overlap)
            content_type = req_body.get('content_type', 'text/plain')
            
            activity_logger.set_activity_info("document_id", document_id)
            activity_logger.set_activity_info("filename", filename)
            activity_logger.set_activity_info("text_length", len(text_content))
            
            # Initialize clients
            openai_client = AzureOpenAI(
                api_key=config.openai.api_key,
                api_version=config.openai.api_version,
                azure_endpoint=config.openai.endpoint
            )
            
            search_client = SearchClient(
                endpoint=config.search.endpoint,
                index_name=config.search.index_name,
                credential=AzureKeyCredential(config.search.api_key)
            )
            
            # Step 1: Process document with sophisticated chunking
            logger.info(f"Processing document: {filename} ({len(text_content)} chars)")
            
            processor = DocumentProcessor(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                use_rcts=config.processing.use_rcts,
                encoding_name=config.processing.encoding_name
            )
            
            chunked_document = processor.process_document(text_content, filename, content_type, activity_logger)
            
            if not chunked_document.chunks:
                return func.HttpResponse(
                    json.dumps({"error": "No chunks generated from the text"}),
                    status_code=400,
                    mimetype="application/json"
                )
            
            # Get processing stats
            processing_stats = processor.get_processing_stats(chunked_document)
            log_processing_stats(logger, processing_stats, activity_logger)
            
            # Convert to format for embeddings
            chunks_for_embedding = []
            for i, chunk in enumerate(chunked_document.chunks):
                chunk_data = {
                    "id": str(uuid.uuid4()),
                    "content": chunk.page_content,
                    "chunk_index": i,
                    "metadata": chunk.metadata,
                    "file_extension": Path(filename).suffix.lower(),
                    "title": chunk.metadata.get("source", {}).get("title", filename),
                    "token_count": len(chunk.page_content.split())  # Rough estimate
                }
                chunks_for_embedding.append(chunk_data)
            
            # Step 2: Generate embeddings with batching
            logger.info(f"Generating embeddings for {len(chunks_for_embedding)} chunks")
            embedded_chunks = generate_embeddings_batch(chunks_for_embedding, openai_client, config)
            
            if not embedded_chunks:
                return func.HttpResponse(
                    json.dumps({"error": "No embeddings generated"}),
                    status_code=500,
                    mimetype="application/json"
                )
            
            # Step 3: Update search index with batching
            logger.info(f"Updating search index with {len(embedded_chunks)} chunks")
            index_result = update_search_index_batch(embedded_chunks, search_client, document_id, config)
            
            # Compile comprehensive response
            response = {
                "document_id": document_id,
                "filename": filename,
                "processing_stats": processing_stats,
                "embedding_stats": {
                    "total_chunks": len(chunks_for_embedding),
                    "embedded_chunks": len(embedded_chunks),
                    "embedding_success_rate": len(embedded_chunks) / len(chunks_for_embedding) if chunks_for_embedding else 0
                },
                "index_stats": index_result,
                "metadata": {
                    "content_type": content_type,
                    "processing_config": {
                        "chunk_size": chunk_size,
                        "chunk_overlap": chunk_overlap,
                        "use_rcts": config.processing.use_rcts,
                        "encoding_name": config.processing.encoding_name
                    }
                }
            }
            
            activity_logger.set_activity_info("total_chunks", len(chunks_for_embedding))
            activity_logger.set_activity_info("embedded_chunks", len(embedded_chunks))
            activity_logger.set_activity_info("uploaded_chunks", index_result.get("uploaded_count", 0))
            
            status_code = 200 if index_result["success"] else 207  # 207 for partial success
            
            return func.HttpResponse(
                json.dumps(response, indent=2),
                status_code=status_code,
                mimetype="application/json"
            )
            
        except ValueError as e:
            activity_logger.error(f"Validation error: {str(e)}")
            return func.HttpResponse(
                json.dumps({"error": f"Configuration or validation error: {str(e)}"}),
                status_code=400,
                mimetype="application/json"
            )
        except Exception as e:
            activity_logger.error(f"Processing error: {str(e)}")
            return func.HttpResponse(
                json.dumps({"error": f"Processing failed: {str(e)}"}),
                status_code=500,
                mimetype="application/json"
            )


@app.route(route="health", methods=["GET"])
def health_check(req: func.HttpRequest) -> func.HttpResponse:
    """Health check endpoint"""
    return func.HttpResponse(
        json.dumps({"status": "healthy"}),
        status_code=200,
        mimetype="application/json"
    )