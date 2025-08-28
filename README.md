# Azure Function: Production-Grade RAG Processing

This Azure Function provides **enterprise-grade document processing** with sophisticated chunking, embedding generation, and search indexing capabilities, inspired by and matching the production quality of **AzureML RAG**.

## 🚀 Key Features

### **Sophisticated Text Processing**
- **Format-Aware Chunking**: Different strategies for Python, Markdown, plain text
- **Token-Aware Processing**: Uses tiktoken for precise token counting and budget management  
- **Recursive Character Splitting**: Language-specific separators and intelligent boundary detection
- **Metadata Preservation**: Maintains document structure, headers, and context

### **Production-Ready Architecture**
- **Comprehensive Configuration**: Environment-based config with validation
- **Advanced Error Handling**: Structured logging, activity tracking, and retry logic
- **Batch Processing**: Efficient handling of large documents with configurable batch sizes
- **Performance Monitoring**: Built-in metrics and duration tracking

### **Enterprise Integration**
- **Azure OpenAI**: Advanced embedding generation with batching and retry logic
- **Azure Cognitive Search**: Sophisticated indexing with rich metadata and structure preservation
- **Flexible Input**: Supports multiple content types and file formats

## Setup

### 1. Prerequisites
- Azure Functions Core Tools
- Python 3.8+
- Azure OpenAI resource
- Azure Cognitive Search service

### 2. Configuration
Update `local.settings.json` with your service credentials:

```json
{
  "Values": {
    "OPENAI_API_KEY": "your-openai-api-key",
    "OPENAI_ENDPOINT": "https://your-openai-resource.openai.azure.com/",
    "SEARCH_SERVICE_ENDPOINT": "https://your-search-service.search.windows.net",
    "SEARCH_SERVICE_KEY": "your-search-service-key",
    "SEARCH_INDEX_NAME": "your-index-name"
  }
}
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run Locally
```bash
func start
```

## Usage

### Process Document Endpoint
**POST** `http://localhost:7071/api/process_document`

Request body:
```json
{
  "text": "Your text content here...",
  "document_id": "optional-document-id",
  "chunk_size": 1000,
  "chunk_overlap": 200
}
```

Response:
```json
{
  "document_id": "unique-document-id",
  "total_chunks": 5,
  "embedded_chunks": 5,
  "index_update": {
    "success": true,
    "uploaded_count": 5
  },
  "processing_summary": {
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "text_length": 4523,
    "chunks_generated": 5,
    "chunks_embedded": 5
  }
}
```

### Health Check Endpoint
**GET** `http://localhost:7071/api/health`

## Azure Cognitive Search Index Schema

Your search index should have the following fields:

```json
{
  "fields": [
    {
      "name": "id",
      "type": "Edm.String",
      "key": true,
      "searchable": false
    },
    {
      "name": "document_id",
      "type": "Edm.String",
      "searchable": false,
      "filterable": true
    },
    {
      "name": "content",
      "type": "Edm.String",
      "searchable": true
    },
    {
      "name": "embedding",
      "type": "Collection(Edm.Single)",
      "searchable": true,
      "dimensions": 1536,
      "vectorSearchProfile": "vector-profile"
    },
    {
      "name": "chunk_index",
      "type": "Edm.Int32",
      "filterable": true
    },
    {
      "name": "start_offset",
      "type": "Edm.Int32"
    },
    {
      "name": "end_offset",
      "type": "Edm.Int32"
    }
  ]
}
```

## Deployment

### Using Azure CLI
```bash
func azure functionapp publish <your-function-app-name>
```

### Environment Variables for Production
Set these in your Azure Function App configuration:
- `OPENAI_API_KEY`
- `OPENAI_ENDPOINT`  
- `OPENAI_API_VERSION`
- `OPENAI_EMBEDDING_MODEL`
- `SEARCH_SERVICE_ENDPOINT`
- `SEARCH_SERVICE_KEY`
- `SEARCH_INDEX_NAME`

## Error Handling

The function includes comprehensive error handling for:
- Missing required parameters
- Invalid text content
- OpenAI API failures
- Azure Cognitive Search failures
- Missing environment variables

## Performance Considerations

- **Chunk Size**: Default 1000 characters, adjust based on your needs
- **Batch Processing**: Currently processes one document at a time
- **Timeout**: Set to 5 minutes in `host.json`
- **Rate Limits**: Consider OpenAI API rate limits for large documents