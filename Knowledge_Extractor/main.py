"""
Main entry point for simple usage.
"""
from .config.settings import PipelineConfig
from .pipeline.orchestrator import KnowledgeExtractionPipeline


def extract_knowledge(pdf_path: str, output_dir: str = "output"):
    """
    Simple function to extract knowledge from a PDF.
    
    Args:
        pdf_path: Path to PDF file.
        output_dir: Output directory.
        
    Returns:
        List of ExtractionResult objects.
    """
    config = PipelineConfig(output_dir=output_dir)
    pipeline = KnowledgeExtractionPipeline(config)
    return pipeline.process(pdf_path, output_dir)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python -m Knowledge_Extractor <pdf_path> [output_dir]")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "output"
    
    results = extract_knowledge(pdf_path, output_dir)
    print(f"Extracted {len(results)} chunks")
