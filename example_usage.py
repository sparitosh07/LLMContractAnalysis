"""
Example usage script for the enhanced Azure Function.
"""

import requests
import json

# Configuration
FUNCTION_URL = "http://localhost:7071/api/process_document"

# Sample contract text
CONTRACT_TEXT = """
SOFTWARE LICENSE AGREEMENT

This Software License Agreement ("Agreement") is entered into on [DATE] between Company X ("Licensor") and the customer ("Licensee").

1. GRANT OF LICENSE
Subject to the terms and conditions of this Agreement, Licensor hereby grants to Licensee a non-exclusive, non-transferable license to use the software ("Software") solely for internal business purposes.

2. RESTRICTIONS
Licensee shall not:
(a) Copy, modify, or distribute the Software
(b) Reverse engineer, decompile, or disassemble the Software
(c) Use the Software for any unlawful purposes

3. TERM AND TERMINATION
This Agreement shall commence on the date first written above and shall continue until terminated. Either party may terminate this Agreement at any time with thirty (30) days written notice.

4. WARRANTY DISCLAIMER
THE SOFTWARE IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.

5. LIMITATION OF LIABILITY
IN NO EVENT SHALL LICENSOR BE LIABLE FOR ANY INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THE SOFTWARE.

6. GOVERNING LAW
This Agreement shall be governed by and construed in accordance with the laws of the State of California, without regard to its conflict of law principles.

7. ENTIRE AGREEMENT
This Agreement constitutes the entire agreement between the parties with respect to the subject matter hereof and supersedes all prior or contemporaneous understandings or agreements.

IN WITNESS WHEREOF, the parties have executed this Agreement as of the date first written above.
"""

def test_basic_processing():
    """Test basic document processing."""
    payload = {
        "text": CONTRACT_TEXT,
        "filename": "software_license.txt",
        "document_id": "contract_001",
        "chunk_size": 800,
        "chunk_overlap": 100,
        "content_type": "text/plain"
    }
    
    print("Testing basic document processing...")
    try:
        response = requests.post(FUNCTION_URL, json=payload, timeout=300)
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code in [200, 207]:
            result = response.json()
            print(f"✅ Success! Processed {result['processing_stats']['total_chunks']} chunks")
            print(f"   Embeddings: {result['embedding_stats']['embedded_chunks']}")
            print(f"   Indexed: {result['index_stats']['uploaded_count']}")
            print(f"   Success Rate: {result['embedding_stats']['embedding_success_rate']:.2%}")
        else:
            print(f"❌ Error: {response.text}")
            
    except Exception as e:
        print(f"❌ Request failed: {str(e)}")


def test_markdown_processing():
    """Test Markdown document processing."""
    markdown_text = """# Contract Analysis Report

## Executive Summary
This report analyzes the key terms and conditions of the software license agreement.

### Key Findings
- **License Type**: Non-exclusive, non-transferable
- **Usage Restrictions**: Internal business purposes only
- **Term**: Terminable with 30 days notice

## Detailed Analysis

### Section 1: Grant of License
The license granted is restrictive, limiting usage to internal business purposes only.

### Section 2: Restrictions
The restrictions are standard for software licensing agreements:

1. No copying or distribution
2. No reverse engineering
3. No unlawful use

### Section 3: Termination Clause
The 30-day termination clause provides flexibility for both parties.

## Recommendations
Consider negotiating for:
- Broader usage rights
- Longer termination notice period
- Limited warranty provisions
"""
    
    payload = {
        "text": markdown_text,
        "filename": "contract_analysis.md",
        "document_id": "analysis_001",
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "content_type": "text/markdown"
    }
    
    print("\nTesting Markdown document processing...")
    try:
        response = requests.post(FUNCTION_URL, json=payload, timeout=300)
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code in [200, 207]:
            result = response.json()
            print(f"✅ Success! Processed {result['processing_stats']['total_chunks']} chunks")
            print(f"   File type: {result['processing_stats']['file_extension']}")
            print(f"   Chunk strategy: {'RCTS' if result['metadata']['processing_config']['use_rcts'] else 'Header-based'}")
        else:
            print(f"❌ Error: {response.text}")
            
    except Exception as e:
        print(f"❌ Request failed: {str(e)}")


def test_python_code_processing():
    """Test Python code processing."""
    python_code = '''"""
Advanced text processing utilities for contract analysis.

This module provides sophisticated text processing capabilities
including chunking, embedding generation, and search indexing.
"""

import re
from typing import List, Dict, Any
from dataclasses import dataclass

@dataclass
class ContractSection:
    """Represents a section of a legal contract."""
    title: str
    content: str
    section_number: int
    subsections: List[str] = None

class ContractProcessor:
    """Process legal contracts for analysis."""
    
    def __init__(self, chunk_size: int = 1000):
        """Initialize the contract processor."""
        self.chunk_size = chunk_size
        self.patterns = {
            'section_header': r'^\\d+\\.',
            'subsection': r'^\\([a-z]\\)',
            'clause': r'^[A-Z ]+:',
        }
    
    def extract_sections(self, text: str) -> List[ContractSection]:
        """Extract sections from contract text."""
        sections = []
        current_section = None
        
        for line in text.split('\\n'):
            if re.match(self.patterns['section_header'], line):
                if current_section:
                    sections.append(current_section)
                current_section = ContractSection(
                    title=line.strip(),
                    content="",
                    section_number=len(sections) + 1
                )
            elif current_section:
                current_section.content += line + "\\n"
        
        if current_section:
            sections.append(current_section)
            
        return sections
    
    def analyze_terms(self, sections: List[ContractSection]) -> Dict[str, Any]:
        """Analyze contract terms and conditions."""
        analysis = {
            'total_sections': len(sections),
            'key_terms': [],
            'restrictions': [],
            'obligations': []
        }
        
        for section in sections:
            # Extract key terms using pattern matching
            content_lower = section.content.lower()
            
            if 'shall not' in content_lower or 'prohibited' in content_lower:
                analysis['restrictions'].append(section.title)
            
            if 'shall' in content_lower or 'must' in content_lower:
                analysis['obligations'].append(section.title)
        
        return analysis

def main():
    """Main processing function."""
    processor = ContractProcessor(chunk_size=800)
    
    # Process contract sections
    with open('contract.txt', 'r') as f:
        contract_text = f.read()
    
    sections = processor.extract_sections(contract_text)
    analysis = processor.analyze_terms(sections)
    
    print(f"Processed {len(sections)} sections")
    print(f"Found {len(analysis['restrictions'])} restrictions")
    print(f"Found {len(analysis['obligations'])} obligations")

if __name__ == "__main__":
    main()
'''
    
    payload = {
        "text": python_code,
        "filename": "contract_processor.py",
        "document_id": "code_001",
        "chunk_size": 1200,
        "chunk_overlap": 150,
        "content_type": "text/x-python"
    }
    
    print("\nTesting Python code processing...")
    try:
        response = requests.post(FUNCTION_URL, json=payload, timeout=300)
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code in [200, 207]:
            result = response.json()
            print(f"✅ Success! Processed {result['processing_stats']['total_chunks']} chunks")
            print(f"   Language-aware chunking applied")
            print(f"   Total tokens: {result['processing_stats']['total_tokens']}")
        else:
            print(f"❌ Error: {response.text}")
            
    except Exception as e:
        print(f"❌ Request failed: {str(e)}")


def test_health_endpoint():
    """Test health check endpoint."""
    print("\nTesting health endpoint...")
    try:
        response = requests.get("http://localhost:7071/api/health", timeout=30)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            print("✅ Health check passed")
        else:
            print(f"❌ Health check failed: {response.text}")
            
    except Exception as e:
        print(f"❌ Health check failed: {str(e)}")


if __name__ == "__main__":
    print("🚀 Testing Enhanced Azure Function with AzureML RAG-style Processing")
    print("=" * 70)
    
    # Run tests
    test_health_endpoint()
    test_basic_processing()
    test_markdown_processing() 
    test_python_code_processing()
    
    print("\n" + "=" * 70)
    print("✨ Testing completed!")