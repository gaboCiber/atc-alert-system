"""
Command-line interface for Knowledge Extractor.
"""
import argparse
import sys
from pathlib import Path

from .config.settings import PipelineConfig, ModelConfig, EmbeddingConfig
from .pipeline.orchestrator import KnowledgeExtractionPipeline


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Extract aeronautical knowledge from PDF documents."
    )
    
    parser.add_argument(
        "pdf_path",
        help="Path to PDF document to process"
    )
    
    parser.add_argument(
        "-o", "--output",
        default="output",
        help="Output directory (default: output)"
    )
    
    parser.add_argument(
        "-m", "--model",
        default="llama3.2",
        help="LLM model name (default: llama3.2)"
    )
    
    parser.add_argument(
        "-g", "--granularity",
        choices=["page", "sentence", "chunk"],
        default="sentence",
        help="Processing granularity: page (full page), sentence (individual sentences), chunk (logical chunks with LLM) (default: sentence)"
    )
    
    parser.add_argument(
        "--base-url",
        default="http://localhost:11434/v1",
        help="Ollama API base URL"
    )
    
    parser.add_argument(
        "--margins",
        nargs=4,
        type=float,
        metavar=("LEFT", "BOTTOM", "RIGHT", "TOP"),
        help="PDF margins in points: left bottom right top"
    )
    
    args = parser.parse_args()
    
    # Validate PDF exists
    if not Path(args.pdf_path).exists():
        print(f"Error: PDF file not found: {args.pdf_path}")
        sys.exit(1)
    
    # Build config
    model_config = ModelConfig(
        name=args.model,
        base_url=args.base_url,
    )
    
    config = PipelineConfig(
        model=model_config,
        granularity=args.granularity,
        output_dir=args.output,
        margins=tuple(args.margins) if args.margins else None,
    )
    
    # Run pipeline
    print(f"Processing: {args.pdf_path}")
    print(f"Model: {args.model}")
    print(f"Granularity: {args.granularity}")
    print(f"Output: {args.output}")
    
    pipeline = KnowledgeExtractionPipeline(config)
    
    try:
        results = pipeline.process(args.pdf_path, args.output)
        
        # Summary
        successful = sum(1 for r in results if r.extraction is not None)
        failed = sum(1 for r in results if r.extraction is None)
        
        print(f"\nExtraction complete!")
        print(f"  Total chunks: {len(results)}")
        print(f"  Successful: {successful}")
        print(f"  Failed: {failed}")
        print(f"  Total entities accumulated: {pipeline.context_manager.get_entity_count()}")
        
    except Exception as e:
        print(f"Error during extraction: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
