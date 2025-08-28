"""Advanced text splitters similar to AzureML RAG implementation."""

import copy
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List, Optional, Callable, Dict
from utils.tokens import token_length_function, tiktoken_cache_dir


@dataclass
class Document:
    """Document with content and metadata."""
    page_content: str
    metadata: Dict[str, Any]
    document_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert document to dictionary."""
        return {
            "content": self.page_content,
            "metadata": self.metadata,
            "document_id": self.document_id
        }


class TextSplitter(ABC):
    """Abstract base class for text splitters."""
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        length_function: Optional[Callable[[str], int]] = None,
        **kwargs
    ):
        """Initialize text splitter."""
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._length_function = length_function or len
        
    @abstractmethod
    def split_text(self, text: str) -> List[str]:
        """Split text into chunks."""
        pass
        
    def create_documents(
        self, 
        texts: List[str], 
        metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> List[Document]:
        """Create documents from texts."""
        _metadatas = metadatas or [{}] * len(texts)
        documents = []
        
        for i, text in enumerate(texts):
            for chunk in self.split_text(text):
                doc = Document(
                    page_content=chunk,
                    metadata=copy.deepcopy(_metadatas[i])
                )
                documents.append(doc)
                
        return documents
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into smaller chunks."""
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        return self.create_documents(texts, metadatas)


class RecursiveCharacterTextSplitter(TextSplitter):
    """Recursively split text using different separators."""
    
    def __init__(
        self,
        separators: Optional[List[str]] = None,
        is_separator_regex: bool = False,
        **kwargs
    ):
        """Initialize recursive text splitter."""
        super().__init__(**kwargs)
        self._separators = separators or ["\n\n", "\n", " ", ""]
        self._is_separator_regex = is_separator_regex
        
    @classmethod
    def from_tiktoken_encoder(
        cls,
        encoding_name: str = "cl100k_base",
        separators: Optional[List[str]] = None,
        **kwargs
    ) -> "RecursiveCharacterTextSplitter":
        """Create splitter with tiktoken encoder."""
        with tiktoken_cache_dir():
            return cls(
                separators=separators,
                length_function=token_length_function(encoding_name),
                **kwargs
            )
    
    @staticmethod
    def get_separators_for_language(language: str) -> List[str]:
        """Get separators for specific programming language."""
        if language.lower() == "python":
            return [
                # Class and function definitions
                r"\nclass ",
                r"\ndef ",
                r"\n\s*def ",
                r"\n\s*async def ",
                # Control structures
                r"\nif ",
                r"\nfor ",
                r"\nwhile ",
                r"\ntry:",
                r"\nwith ",
                # Comments and docstrings
                '"""',
                "'''",
                r"\n# ",
                # Basic separators
                "\n\n",
                "\n",
                " ",
                "",
            ]
        else:
            return ["\n\n", "\n", " ", ""]
    
    def split_text(self, text: str) -> List[str]:
        """Split text recursively using separators."""
        return self._split_text(text, self._separators)
        
    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        """Recursively split text."""
        final_chunks = []
        separator = separators[-1] if separators else ""
        new_separators = []
        
        for i, s in enumerate(separators):
            _separator = s if not self._is_separator_regex else s
            if _separator == "":
                separator = s
                break
            if re.search(_separator if self._is_separator_regex else re.escape(s), text):
                separator = s
                new_separators = separators[i + 1:]
                break
                
        splits = self._split_text_with_regex(text, separator, self._is_separator_regex)
        
        good_splits = []
        _separator = separator if not self._is_separator_regex else ""
        
        for s in splits:
            if self._length_function(s) < self._chunk_size:
                good_splits.append(s)
            else:
                if good_splits:
                    merged_text = self._merge_splits(good_splits, _separator)
                    final_chunks.extend(merged_text)
                    good_splits = []
                
                if not new_separators:
                    final_chunks.append(s)
                else:
                    other_info = self._split_text(s, new_separators)
                    final_chunks.extend(other_info)
        
        if good_splits:
            merged_text = self._merge_splits(good_splits, _separator)
            final_chunks.extend(merged_text)
            
        return final_chunks
    
    def _split_text_with_regex(
        self, 
        text: str, 
        separator: str, 
        is_separator_regex: bool
    ) -> List[str]:
        """Split text with regex or string separator."""
        if separator:
            if is_separator_regex:
                splits = re.split(separator, text)
            else:
                splits = text.split(separator)
        else:
            splits = list(text)
            
        return [s for s in splits if s != ""]
    
    def _merge_splits(self, splits: List[str], separator: str) -> List[str]:
        """Merge splits into chunks with overlap."""
        if not splits:
            return []
            
        docs = []
        current_doc = []
        total = 0
        
        for split in splits:
            split_len = self._length_function(split)
            
            if total + split_len + (len(current_doc) * len(separator) if separator else 0) > self._chunk_size and current_doc:
                doc = separator.join(current_doc).strip() if separator else "".join(current_doc)
                if doc:
                    docs.append(doc)
                
                while total > self._chunk_overlap or (total + split_len + (len(current_doc) * len(separator) if separator else 0) > self._chunk_size and total > 0):
                    if current_doc:
                        total -= self._length_function(current_doc[0]) + (len(separator) if separator else 0)
                        current_doc = current_doc[1:]
                    else:
                        break
                        
            current_doc.append(split)
            total += split_len + (len(separator) if separator else 0)
            
        doc = separator.join(current_doc).strip() if separator else "".join(current_doc)
        if doc:
            docs.append(doc)
            
        return docs


class TokenTextSplitter(TextSplitter):
    """Split text based on token count."""
    
    def __init__(
        self,
        encoding_name: str = "cl100k_base",
        allowed_special: set = set(),
        disallowed_special: set = set(),
        **kwargs
    ):
        """Initialize token text splitter."""
        try:
            import tiktoken
            self._tokenizer = tiktoken.get_encoding(encoding_name)
        except ImportError:
            raise ImportError("tiktoken is required for TokenTextSplitter")
            
        self._allowed_special = allowed_special
        self._disallowed_special = disallowed_special
        
        super().__init__(
            length_function=token_length_function(encoding_name),
            **kwargs
        )
    
    def split_text(self, text: str) -> List[str]:
        """Split text by tokens."""
        def _encode(_text: str):
            return self._tokenizer.encode(
                _text,
                allowed_special=self._allowed_special,
                disallowed_special=self._disallowed_special,
            )
        
        tokenized = _encode(text)
        chunks = []
        
        for i in range(0, len(tokenized), self._chunk_size - self._chunk_overlap):
            chunk_tokens = tokenized[i : i + self._chunk_size]
            chunk_text = self._tokenizer.decode(chunk_tokens)
            chunks.append(chunk_text)
            
        return chunks


class MarkdownHeaderTextSplitter(TextSplitter):
    """Split markdown text preserving header hierarchy."""
    
    def __init__(self, remove_hyperlinks: bool = True, remove_images: bool = True, **kwargs):
        """Initialize markdown header splitter."""
        self._remove_hyperlinks = remove_hyperlinks
        self._remove_images = remove_images
        
        # Create sub-splitter for large sections
        self._sub_splitter = TokenTextSplitter(encoding_name="cl100k_base", **kwargs)
        super().__init__(**kwargs)
    
    @classmethod
    def from_tiktoken_encoder(
        cls,
        encoding_name: str = "cl100k_base",
        **kwargs
    ) -> "MarkdownHeaderTextSplitter":
        """Create markdown splitter with tiktoken encoder."""
        with tiktoken_cache_dir():
            return cls(
                length_function=token_length_function(encoding_name),
                **kwargs
            )
    
    def split_text(self, text: str) -> List[str]:
        """Split markdown text by headers."""
        blocks = self._parse_markdown_blocks(text)
        chunks = []
        
        for block in blocks:
            nested_headers = self._get_nested_headers(block)
            content = f"{nested_headers}\n{block['content']}" if nested_headers else block['content']
            
            if self._length_function(content) > self._chunk_size:
                # Split large sections
                sub_chunks = self._sub_splitter.split_text(block['content'])
                for sub_chunk in sub_chunks:
                    full_chunk = f"{nested_headers}\n{sub_chunk}" if nested_headers else sub_chunk
                    chunks.append(full_chunk)
            else:
                chunks.append(content)
                
        return chunks
    
    def _parse_markdown_blocks(self, text: str) -> List[Dict[str, Any]]:
        """Parse markdown into blocks with header hierarchy."""
        # Remove hyperlinks and images if requested
        if self._remove_hyperlinks:
            text = re.sub(r'\[([^\]]*)\]\([^\)]*\)', r'\1', text)
        if self._remove_images:
            text = re.sub(r'!\[([^\]]*)\]\([^\)]*\)', '', text)
            
        # Split by headers
        blocks = re.split(r'^(#+\s.*?)$', text, flags=re.MULTILINE)
        blocks = [b.strip() for b in blocks if b.strip()]
        
        parsed_blocks = []
        header_stack = []
        
        if not blocks[0].startswith('#'):
            # Content before first header
            parsed_blocks.append({
                'header': None,
                'content': blocks[0],
                'level': 0,
                'parents': []
            })
            blocks = blocks[1:]
        
        for i in range(0, len(blocks), 2):
            if i >= len(blocks):
                break
                
            header = blocks[i]
            content = blocks[i + 1] if i + 1 < len(blocks) else ""
            
            level = header.count('#')
            
            # Update header stack
            while header_stack and header_stack[-1]['level'] >= level:
                header_stack.pop()
                
            parents = [h['header'] for h in header_stack]
            header_stack.append({'header': header, 'level': level})
            
            parsed_blocks.append({
                'header': header,
                'content': content,
                'level': level,
                'parents': parents
            })
            
        return parsed_blocks
    
    def _get_nested_headers(self, block: Dict[str, Any]) -> str:
        """Get nested header string for a block."""
        headers = []
        if block['parents']:
            headers.extend(block['parents'])
        if block['header']:
            headers.append(block['header'])
            
        return '\n'.join(headers) if headers else ""


# Factory function for getting appropriate splitter
def get_text_splitter(
    file_extension: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    use_rcts: bool = True,
    use_nltk: bool = False,
    **kwargs
) -> TextSplitter:
    """Get appropriate text splitter for file extension."""
    
    # Handle code files
    if file_extension == ".py":
        with tiktoken_cache_dir():
            return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                encoding_name="cl100k_base",
                separators=RecursiveCharacterTextSplitter.get_separators_for_language("python"),
                is_separator_regex=True,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                disallowed_special=(),
                **kwargs
            )
    
    # Handle markdown files
    elif file_extension == ".md":
        if use_rcts:
            with tiktoken_cache_dir():
                return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                    encoding_name="cl100k_base",
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    disallowed_special=(),
                    **kwargs
                )
        else:
            with tiktoken_cache_dir():
                return MarkdownHeaderTextSplitter.from_tiktoken_encoder(
                    encoding_name="cl100k_base",
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    disallowed_special=(),
                    **kwargs
                )
    
    # Handle text and other formats
    else:
        with tiktoken_cache_dir():
            return TokenTextSplitter(
                encoding_name="cl100k_base",
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                disallowed_special=(),
                **kwargs
            )