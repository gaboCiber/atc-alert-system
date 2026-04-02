"""
Command-line interface for Knowledge Extractor.
"""
import argparse
import sys
from pathlib import Path

from .config.settings import PipelineConfig, ModelConfig, EmbeddingConfig, ResumeConfig
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
    
    # Resume options
    parser.add_argument(
        "--start-page",
        type=int,
        default=1,
        help="Page to start processing from (1-indexed, default: 1)"
    )
    
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from previous run, loading entities from existing output"
    )
    
    parser.add_argument(
        "--previous-dir",
        help="Directory with previous extraction results (for resuming)"
    )
    
    # Chunks source options
    parser.add_argument(
        "--chunks-source",
        help="Directory with pre-generated chunk JSON files (pagina_N_chunks.json). If chunks exist for a page, they will be used instead of extracting from PDF."
    )
    
    parser.add_argument(
        "--chunk-only",
        action="store_true",
        help="Extract and save chunks only, skip KEX extraction"
    )
    
    # Context control options
    parser.add_argument(
        "--no-definitions",
        action="store_true",
        help="Exclude definitions from context"
    )
    
    parser.add_argument(
        "--no-rules",
        action="store_true",
        help="Exclude rules from context"
    )
    
    parser.add_argument(
        "--no-relationships",
        action="store_true",
        help="Exclude relationships from context"
    )
    
    parser.add_argument(
        "--definition-limit",
        type=int,
        default=10,
        help="Maximum definitions to include in context (default: 10)"
    )
    
    parser.add_argument(
        "--rule-limit",
        type=int,
        default=5,
        help="Maximum rules to include in context (default: 5)"
    )
    
    parser.add_argument(
        "--relationship-limit",
        type=int,
        default=10,
        help="Maximum relationships to include in context (default: 10)"
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
    
    embedding_config = EmbeddingConfig(
        definition_top_k=args.definition_limit,
        rule_top_k=args.rule_limit,
        relationship_top_k=args.relationship_limit,
        include_definitions=not args.no_definitions,
        include_rules=not args.no_rules,
        include_relationships=not args.no_relationships,
    )
    
    resume_config = ResumeConfig(
        start_page=args.start_page,
        load_previous_entities=args.resume or args.start_page > 1,
        previous_output_dir=args.previous_dir,
    )
    
    # Strip quotes from chunks_source if present
    chunks_source = args.chunks_source
    if chunks_source:
        chunks_source = chunks_source.strip('"\'')
    
    config = PipelineConfig(
        model=model_config,
        embedding=embedding_config,
        granularity=args.granularity,
        output_dir=args.output,
        margins=tuple(args.margins) if args.margins else None,
        resume=resume_config,
        chunks_source_dir=chunks_source,
        chunk_only=args.chunk_only,
    )
    
    # Run pipeline
    if args.start_page > 1:
        print(f"Starting from page: {args.start_page}")
    if args.resume:
        print(f"Resuming from previous state")
    
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
        print(f"  Entities accumulated: {pipeline.context_manager.get_entity_count()}")
        print(f"  Definitions accumulated: {pipeline.context_manager.get_definition_count()}")
        print(f"  Rules accumulated: {pipeline.context_manager.get_rule_count()}")
        print(f"  Relationships accumulated: {pipeline.context_manager.get_relationship_count()}")
        
    except Exception as e:
        print(f"Error during extraction: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
