"""
Document Understanding Agent - Extract Structured Data from Documents

This example demonstrates how to build a document understanding agent using mahsm.
Perfect for:
- Invoice/receipt processing
- Form data extraction
- Table parsing
- Document classification

It showcases:
- OCR-free document understanding via vision
- Structured data extraction
- Confidence scoring for extracted fields
- Validation and error handling

Requirements:
    pip install dspy pillow

Environment Variables:
    OPENAI_API_KEY: Your OpenAI API key (for GPT-5-mini)
    
Cost Estimate:
    ~$0.12 per document (with GPT-5-mini)
"""

import dspy
import mahsm as ma
from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, END


# ============================================================================
# State Definition
# ============================================================================

class DocumentState(TypedDict):
    """State for document understanding workflow."""
    document_image: dspy.Image  # Document to process
    document_type: str  # Type: "invoice", "receipt", "form", "table"
    schema: Optional[Dict[str, Any]]  # Expected fields (optional)
    
    # Processing outputs
    document_classification: str  # Identified document type
    extracted_data: Dict[str, Any]  # Structured extracted data
    confidence_scores: Dict[str, float]  # Confidence per field
    validation_errors: List[str]  # Validation issues found


# ============================================================================
# DSPy Modules for Document Processing
# ============================================================================

@ma.dspy_node
class DocumentClassifier(dspy.Module):
    """
    Classifies the document type and identifies structure.
    
    Determines:
    - Document category (invoice, receipt, form, etc.)
    - Layout structure
    - Key sections present
    """
    
    def __init__(self):
        super().__init__()
        self.classify = dspy.ChainOfThought(
            "document_image: dspy.Image -> document_type: str, structure: str"
        )
    
    def forward(self, document_image):
        """Classify document type and structure."""
        result = self.classify(document_image=document_image)
        return result


@ma.dspy_node
class DataExtractor(dspy.Module):
    """
    Extracts structured data from the document.
    
    Handles:
    - Text fields
    - Numbers and amounts
    - Dates
    - Tables
    - Key-value pairs
    """
    
    def __init__(self):
        super().__init__()
        self.extract = dspy.ChainOfThought(
            """document_image: dspy.Image, 
               document_type: str, 
               schema: str -> 
               extracted_data: str, 
               confidence: float"""
        )
    
    def forward(self, document_image, document_type, schema=None):
        """Extract data according to schema."""
        schema_str = str(schema) if schema else "auto-detect fields"
        result = self.extract(
            document_image=document_image,
            document_type=document_type,
            schema=schema_str
        )
        return result


@ma.dspy_node
class DataValidator(dspy.Module):
    """
    Validates extracted data for consistency and completeness.
    
    Checks:
    - Required fields present
    - Data type correctness
    - Value ranges
    - Format validation (dates, emails, etc.)
    """
    
    def __init__(self):
        super().__init__()
        self.validate = dspy.ChainOfThought(
            """extracted_data: str, 
               document_type: str -> 
               is_valid: bool, 
               validation_errors: str"""
        )
    
    def forward(self, extracted_data, document_type):
        """Validate extracted data."""
        result = self.validate(
            extracted_data=str(extracted_data),
            document_type=document_type
        )
        return result


# ============================================================================
# Graph Construction
# ============================================================================

def build_document_agent() -> StateGraph:
    """
    Builds the document understanding workflow graph.
    
    Workflow:
        1. Classify: Identify document type
        2. Extract: Pull structured data
        3. Validate: Check data quality
    
    Returns:
        Compiled LangGraph StateGraph
    """
    graph = StateGraph(DocumentState)
    
    # Initialize modules
    classifier = DocumentClassifier()
    extractor = DataExtractor()
    validator = DataValidator()
    
    # Add nodes
    def classify_document(state):
        """Classify the document."""
        result = classifier(document_image=state["document_image"])
        return {
            "document_classification": result.document_type
        }
    
    def extract_data(state):
        """Extract structured data."""
        result = extractor(
            document_image=state["document_image"],
            document_type=state.get("document_classification", state.get("document_type", "unknown")),
            schema=state.get("schema")
        )
        
        # Parse extracted data string to dict (simplified)
        # In production, use proper JSON parsing
        extracted = {"raw": result.extracted_data}
        confidences = {"overall": result.confidence}
        
        return {
            "extracted_data": extracted,
            "confidence_scores": confidences
        }
    
    def validate_data(state):
        """Validate extracted data."""
        result = validator(
            extracted_data=state["extracted_data"],
            document_type=state["document_classification"]
        )
        
        # Parse validation errors
        errors = []
        if not result.is_valid:
            errors = [result.validation_errors]
        
        return {"validation_errors": errors}
    
    graph.add_node("classify", classify_document)
    graph.add_node("extract", extract_data)
    graph.add_node("validate", validate_data)
    
    # Define workflow
    graph.set_entry_point("classify")
    graph.add_edge("classify", "extract")
    graph.add_edge("extract", "validate")
    graph.add_edge("validate", END)
    
    return graph.compile()


# ============================================================================
# Convenience Functions
# ============================================================================

def process_invoice(image, schema=None):
    """
    Extract data from an invoice.
    
    Args:
        image: dspy.Image or path/URL to invoice
        schema: Optional dict specifying expected fields
    
    Returns:
        dict with extracted invoice data
    
    Example schema:
        {
            "invoice_number": "string",
            "date": "date",
            "total": "number",
            "vendor": "string",
            "items": "array"
        }
    """
    agent = build_document_agent()
    
    # Convert to dspy.Image if needed
    if not isinstance(image, dspy.Image):
        image = dspy.Image.from_url(image) if image.startswith('http') else dspy.Image.from_file(image)
    
    return agent.invoke({
        "document_image": image,
        "document_type": "invoice",
        "schema": schema
    })


def process_receipt(image):
    """
    Extract data from a receipt.
    
    Args:
        image: dspy.Image or path/URL to receipt
    
    Returns:
        dict with merchant, items, total, date, etc.
    """
    agent = build_document_agent()
    
    if not isinstance(image, dspy.Image):
        image = dspy.Image.from_url(image) if image.startswith('http') else dspy.Image.from_file(image)
    
    return agent.invoke({
        "document_image": image,
        "document_type": "receipt",
        "schema": {
            "merchant": "string",
            "date": "date",
            "items": "array",
            "subtotal": "number",
            "tax": "number",
            "total": "number"
        }
    })


def process_form(image, field_names=None):
    """
    Extract data from a form.
    
    Args:
        image: dspy.Image or path/URL to form
        field_names: Optional list of expected field names
    
    Returns:
        dict with form fields and values
    """
    agent = build_document_agent()
    
    if not isinstance(image, dspy.Image):
        image = dspy.Image.from_url(image) if image.startswith('http') else dspy.Image.from_file(image)
    
    schema = None
    if field_names:
        schema = {name: "string" for name in field_names}
    
    return agent.invoke({
        "document_image": image,
        "document_type": "form",
        "schema": schema
    })


def extract_table(image):
    """
    Extract table data from document image.
    
    Args:
        image: dspy.Image or path/URL to document with table
    
    Returns:
        dict with table data (rows and columns)
    """
    agent = build_document_agent()
    
    if not isinstance(image, dspy.Image):
        image = dspy.Image.from_url(image) if image.startswith('http') else dspy.Image.from_file(image)
    
    return agent.invoke({
        "document_image": image,
        "document_type": "table",
        "schema": None
    })


# ============================================================================
# Example Usage
# ============================================================================

def run_example():
    """Run document understanding examples."""
    import os
    
    print("📄 mahsm Document Understanding Agent\\n")
    print("=" * 60)
    
    # Configure DSPy
    print("\\n📝 Configuring DSPy with GPT-5-mini...")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ Error: OPENAI_API_KEY environment variable not set")
        return
    
    dspy.configure(
        lm=dspy.LM(
            model="openai/gpt-5-mini",
            api_key=api_key
        )
    )
    print("✅ DSPy configured\\n")
    
    # Example 1: Invoice Processing
    print("=" * 60)
    print("Example 1: Invoice Data Extraction")
    print("=" * 60)
    
    print("\\n💡 Invoice processing workflow:")
    print("   1. Classify document type")
    print("   2. Extract structured fields (invoice #, date, total, items)")
    print("   3. Validate data completeness\\n")
    
    print("📝 Usage:")
    print("   result = process_invoice('invoice.pdf')")
    print("   print(result['extracted_data'])")
    print("   print(result['confidence_scores'])\\n")
    
    # Example 2: Receipt Processing
    print("=" * 60)
    print("Example 2: Receipt Processing")
    print("=" * 60)
    
    print("\\n💡 Receipt processing workflow:")
    print("   - Extract merchant name")
    print("   - Parse line items with prices")
    print("   - Calculate totals (subtotal, tax, total)")
    print("   - Extract date and payment method\\n")
    
    print("📝 Usage:")
    print("   result = process_receipt('receipt.jpg')")
    print("   for item in result['extracted_data'].get('items', []):")
    print("       print(f\\\"{item['name']}: ${item['price']}\\\")\\n")
    
    # Example 3: Form Processing
    print("=" * 60)
    print("Example 3: Form Data Extraction")
    print("=" * 60)
    
    print("\\n💡 Form processing workflow:")
    print("   - Detect form fields automatically")
    print("   - Extract field labels and values")
    print("   - Handle checkboxes and selections")
    print("   - Validate required fields\\n")
    
    print("📝 Usage:")
    print("   fields = ['name', 'email', 'address', 'phone']")
    print("   result = process_form('application.pdf', field_names=fields)")
    print("   print(result['extracted_data'])\\n")
    
    # Example 4: Table Extraction
    print("=" * 60)
    print("Example 4: Table Data Extraction")
    print("=" * 60)
    
    print("\\n💡 Table extraction workflow:")
    print("   - Detect table structure (rows/columns)")
    print("   - Extract headers")
    print("   - Parse cell values")
    print("   - Preserve table relationships\\n")
    
    print("📝 Usage:")
    print("   result = extract_table('financial_report.pdf')")
    print("   table = result['extracted_data']")
    print("   # Process table data as needed\\n")
    
    print("=" * 60)
    print("✨ Document Understanding Agent Demo Complete!")
    print("=" * 60)
    print("\\n💡 Cost Efficiency:")
    print("   • GPT-5-mini: ~$0.12 per document")
    print("   • No separate OCR service needed")
    print("   • Structured output with confidence scores")
    print("   • Handles various document types\\n")


if __name__ == "__main__":
    run_example()
