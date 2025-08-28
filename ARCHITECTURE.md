# Architecture: Azure Function with LangChain Integration

This Azure Function now uses **industry-standard LangChain components** for text splitting, exactly matching where **AzureML RAG** uses LangChain, while keeping Azure Cognitive Search for vector storage.

## 🏗️ Architecture Overview

```
Text Input → Document Processing → LangChain Text Splitting → Embedding → ACS Indexing
```

## 📦 LangChain Integration (Matching AzureML RAG)

### **Text Splitters Used**
Based on analysis of AzureML RAG code, we use these **exact LangChain splitters**:

#### **Python Code** (`.py` files)
```python
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    encoding_name="cl100k_base",
    separators=RecursiveCharacterTextSplitter.get_separators_for_language(Language.PYTHON),
    is_separator_regex=True,
    disallowed_special=()
)
```

#### **Text Files** (`.txt`, `.pdf`, `.docx`, etc.)
```python
from langchain_text_splitters import TokenTextSplitter

TokenTextSplitter(
    encoding_name="cl100k_base",
    length_function=token_length_function(),
    disallowed_special=()
)
```

#### **Markdown Files** (`.md`)
```python
from langchain_text_splitters import MarkdownTextSplitter

# Option 1: Standard markdown splitting with RCTS
MarkdownTextSplitter.from_tiktoken_encoder(
    encoding_name="cl100k_base",
    disallowed_special=()
)

# Option 2: Header-aware splitting (custom implementation matching AzureML RAG)
MarkdownHeaderTextSplitter.from_tiktoken_encoder(
    encoding_name="cl100k_base",
    remove_hyperlinks=True,
    remove_images=True
)
```

#### **Optional: NLTK Sentence Splitting**
```python
from langchain_text_splitters import NLTKTextSplitter

NLTKTextSplitter(
    length_function=token_length_function()
)
```

## 🔄 Component Breakdown

### **What Uses LangChain** ✅
- **Text Splitters**: `RecursiveCharacterTextSplitter`, `TokenTextSplitter`, `MarkdownTextSplitter`, `NLTKTextSplitter`
- **Language Detection**: `Language.PYTHON` for code-aware splitting
- **Token Management**: tiktoken integration through LangChain

### **What Stays Custom/Azure-Native** ✅
- **Vector Store**: Azure Cognitive Search (not LangChain vector store)
- **Embeddings Client**: Direct Azure OpenAI client (not LangChain embeddings)
- **Document Loading**: Custom loaders for title extraction and metadata
- **Configuration**: Azure Function-specific config management
- **Activity Logging**: Azure Function monitoring and telemetry

## 📁 File Structure

```
/
├── function_app.py                 # Main Azure Function (uses ACS directly)
├── langchain_text_splitters.py    # LangChain text splitters (industry standard)
├── document_processing.py         # Document processing pipeline  
├── config.py                      # Azure-specific configuration
├── utils/
│   ├── tokens.py                  # Token counting utilities
│   └── logging_utils.py           # Azure Function logging
└── requirements.txt               # LangChain + Azure dependencies
```

## 🎯 Processing Pipeline

### **1. Document Intake**
```python
# Custom document loaders (not LangChain)
loader = TextFileLoader(content, source, metadata)
chunked_doc = loader.load_chunked_document()
```

### **2. LangChain Text Splitting**
```python
# Use industry-standard LangChain splitters
splitter_args = {
    "chunk_size": chunk_size,
    "chunk_overlap": chunk_overlap,
    "use_rcts": use_rcts
}
chunks = split_documents_with_langchain(documents, splitter_args)
```

### **3. Azure OpenAI Embedding**
```python
# Direct Azure OpenAI client (not LangChain wrapper)
openai_client = AzureOpenAI(...)
response = openai_client.embeddings.create(...)
```

### **4. Azure Cognitive Search Indexing**
```python
# Direct ACS client (not LangChain vector store)
search_client = SearchClient(...)
search_client.upload_documents(documents)
```

## 🔍 Comparison with AzureML RAG

| Component | AzureML RAG | Our Implementation |
|-----------|-------------|-------------------|
| **Text Splitting** | LangChain (vendored) | LangChain (standard library) |
| **Python Code** | `RecursiveCharacterTextSplitter` | ✅ Same |
| **Markdown** | `MarkdownTextSplitter` + Custom | ✅ Same |  
| **Text Files** | `TokenTextSplitter` | ✅ Same |
| **Token Counting** | tiktoken + LangChain | ✅ Same |
| **Vector Storage** | Various (ACS, Faiss, etc.) | Azure Cognitive Search |
| **Embeddings** | Azure OpenAI direct | ✅ Same approach |

## 🚀 Benefits of This Architecture

### **Industry Standard** 
- Uses **proven LangChain text splitters** instead of custom implementations
- Maintains compatibility with LangChain ecosystem
- Easier to maintain and update

### **Azure-Optimized**
- **Direct ACS integration** for optimal performance
- **Azure Function-native** configuration and logging  
- **Production-ready** error handling and monitoring

### **Flexibility**
- Easy to **swap text splitters** by changing LangChain components
- **Configurable splitting strategies** via environment variables
- **Extensible** for new document formats

## 🛠️ Configuration

### **Environment Variables**
```bash
# Azure Services (required)
OPENAI_API_KEY=your-openai-key
OPENAI_ENDPOINT=https://your-openai.openai.azure.com/
SEARCH_SERVICE_ENDPOINT=https://your-search.search.windows.net
SEARCH_SERVICE_KEY=your-search-key
SEARCH_INDEX_NAME=your-index

# Text Processing (optional)
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
USE_RCTS=true          # Use recursive character text splitter for markdown
USE_NLTK=false         # Use NLTK sentence splitting
```

This architecture provides the **best of both worlds**: industry-standard LangChain text processing with Azure-native vector storage and embedding generation.