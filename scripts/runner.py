#!/usr/bin/env python
"""
Pipeline Runner - Orchestrates the entire Grid Compliance LLM pipeline.

Usage:
    python runner.py --all           # Run complete pipeline
    python runner.py --process       # Only process PDFs
    python runner.py --generate      # Only generate QA pairs
    python runner.py --train         # Only train the model
    python runner.py --status        # Show pipeline status
"""

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent


def run_command(cmd: list, description: str) -> bool:
    """Run a command and return success status."""
    print(f"\n{'='*60}")
    print(f"🚀 {description}")
    print(f"   Command: {' '.join(cmd)}")
    print('='*60)
    
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    
    if result.returncode == 0:
        print(f"✅ {description} - COMPLETE")
        return True
    else:
        print(f"❌ {description} - FAILED (exit code: {result.returncode})")
        return False


def check_env():
    """Check if environment is properly configured."""
    import os
    from dotenv import load_dotenv
    
    load_dotenv(PROJECT_ROOT / ".env")
    
    issues = []
    
    # Check for Gemini API key
    if not os.getenv("GEMINI_API_KEY"):
        issues.append("GEMINI_API_KEY not set in .env file")
    
    # Check for raw data
    raw_dir = PROJECT_ROOT / "data" / "raw"
    if not raw_dir.exists():
        issues.append(f"Raw data directory not found: {raw_dir}")
    else:
        pdfs = list(raw_dir.glob("*.pdf"))
        if len(pdfs) < 3:
            issues.append(f"Expected 3 PDFs in {raw_dir}, found {len(pdfs)}")
    
    if issues:
        print("\n⚠️  Environment Issues:")
        for issue in issues:
            print(f"   - {issue}")
        return False
    
    print("✅ Environment check passed")
    return True


def show_status():
    """Show the current pipeline status."""
    import sqlite3
    
    db_path = PROJECT_ROOT / "data" / "pipeline.db"
    
    print("\n" + "="*60)
    print("📊 PIPELINE STATUS")
    print("="*60)
    
    # Check PDFs
    raw_dir = PROJECT_ROOT / "data" / "raw"
    interim_dir = PROJECT_ROOT / "data" / "interim"
    
    print(f"\n📁 Raw PDFs: ", end="")
    if raw_dir.exists():
        pdfs = list(raw_dir.glob("*.pdf"))
        print(f"{len(pdfs)} files")
        for pdf in pdfs:
            print(f"   - {pdf.name}")
    else:
        print("Directory not found")
    
    print(f"\n📁 Sliced PDFs: ", end="")
    if interim_dir.exists():
        sliced = list(interim_dir.glob("sliced_*.pdf"))
        print(f"{len(sliced)} files")
    else:
        print("Directory not found")
    
    # Check database
    print(f"\n💾 Database: ", end="")
    if db_path.exists():
        print(f"Found ({db_path})")
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Chunks
        try:
            cursor.execute("SELECT COUNT(*) FROM document_chunks")
            chunk_count = cursor.fetchone()[0]
            print(f"   - Document chunks: {chunk_count}")
            
            cursor.execute("SELECT source_file, COUNT(*) FROM document_chunks GROUP BY source_file")
            for source, count in cursor.fetchall():
                print(f"     • {source}: {count}")
        except:
            print("   - Document chunks: Table not found")
        
        # QA pairs
        try:
            cursor.execute("SELECT COUNT(*) FROM training_dataset WHERE question != 'N/A'")
            qa_count = cursor.fetchone()[0]
            print(f"   - QA pairs: {qa_count}")
            
            cursor.execute("SELECT tier, COUNT(*) FROM training_dataset GROUP BY tier")
            for tier, count in cursor.fetchall():
                print(f"     • {tier}: {count}")
        except:
            print("   - QA pairs: Table not found")
        
        conn.close()
    else:
        print("Not found")
    
    # Check models
    models_dir = PROJECT_ROOT / "models"
    print(f"\n🤖 Models: ", end="")
    if models_dir.exists() and any(models_dir.iterdir()):
        print("Found")
        for model_dir in models_dir.iterdir():
            if model_dir.is_dir():
                print(f"   - {model_dir.name}")
    else:
        print("Not trained yet")
    
    print("\n" + "="*60)


def main():
    parser = argparse.ArgumentParser(
        description="Grid Compliance LLM Pipeline Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python runner.py --status         # Check current status
  python runner.py --all            # Run complete pipeline
  python runner.py --process        # Process PDFs only
  python runner.py --generate       # Generate QA pairs only
  python runner.py --train          # Train model only
        """
    )
    
    parser.add_argument("--all", action="store_true", help="Run complete pipeline")
    parser.add_argument("--process", action="store_true", help="Process PDFs")
    parser.add_argument("--generate", action="store_true", help="Generate QA pairs")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--status", action="store_true", help="Show pipeline status")
    parser.add_argument("--skip-env-check", action="store_true", help="Skip environment check")
    
    args = parser.parse_args()
    
    # Default to status if no args
    if not any([args.all, args.process, args.generate, args.train, args.status]):
        args.status = True
    
    if args.status:
        show_status()
        return
    
    # Environment check
    if not args.skip_env_check:
        if not check_env():
            print("\n❌ Fix environment issues before running pipeline")
            sys.exit(1)
    
    success = True
    
    # Process PDFs
    if args.all or args.process:
        success = run_command(
            [sys.executable, "src/processing/pdf_processor.py"],
            "Processing PDFs"
        ) and success
    
    # Generate QA pairs
    if args.all or args.generate:
        success = run_command(
            [sys.executable, "src/generation/qa_generator.py", "--generate"],
            "Generating QA Pairs"
        ) and success
        
        # Show stats
        run_command(
            [sys.executable, "src/generation/qa_generator.py", "--stats"],
            "Showing QA Statistics"
        )
        
        # Export to JSONL
        run_command(
            [sys.executable, "src/generation/qa_generator.py", "--export"],
            "Exporting to JSONL"
        )
    
    # Train model
    if args.all or args.train:
        success = run_command(
            [sys.executable, "src/training/trainer.py", "--train"],
            "Training Model"
        ) and success
    
    # Final status
    print("\n" + "="*60)
    if success:
        print("🎉 PIPELINE COMPLETE!")
    else:
        print("⚠️  PIPELINE COMPLETED WITH ERRORS")
    print("="*60)
    
    show_status()


if __name__ == "__main__":
    main()
