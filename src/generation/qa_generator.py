"""
QA Generator - Three-Tier Strategy for Grid Compliance Training Data

Tier 1 (Fact): Hard numerical limits from G99 tables
Tier 2 (Design): Compliance rules from UKPN EDS  
Tier 3 (Scenario): Fleet logistics and project estimation from SPEN Guide
"""

import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Optional

import google.generativeai as genai
from dotenv import load_dotenv
from tqdm import tqdm

# Load environment variables
load_dotenv()

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DB_PATH = PROJECT_ROOT / "data" / "pipeline.db"

# Configure Gemini
api_key = os.getenv("GEMINI_API_KEY")
if api_key:
    genai.configure(api_key=api_key)

# =============================================================================
# Three-Tier Prompt Templates
# =============================================================================

TIER1_PROMPT = """
You are a Senior Grid Compliance Engineer specializing in UK power grid regulations.

### CONTEXT (G99 Engineering Recommendation):
Source: {source} | Page: {page}
---
{content}
---

### TASK:
Generate a FACTUAL Question and Answer pair focused on:
- Specific numerical thresholds (kW, kVA, Hz, voltage levels)
- Equipment classifications (Type A, B, C, D)
- Protection settings and timing requirements
- Frequency response parameters

The question should be something a junior engineer would ask when checking compliance.
The answer must cite the specific numerical value or classification from the text.

### OUTPUT (JSON only):
{{"question": "What is the [specific parameter] for [specific scenario]?", "answer": "[Precise value/classification] as specified in G99 Section X."}}

If the text lacks specific technical data, return: {{"question": "N/A", "answer": "N/A"}}
"""

TIER2_PROMPT = """
You are a Senior Grid Compliance Engineer specializing in UK Distribution Network Operator (DNO) requirements.

### CONTEXT (UKPN EDS 08-5050 - EV Connections):
Source: {source} | Page: {page}
---
{content}
---

### TASK:
Generate a DESIGN/PROCEDURAL Question and Answer pair focused on:
- Connection application processes
- Earthing and safety requirements (PME, TT systems)
- Physical separation rules (2.5m requirement, etc.)
- Metering and notification procedures
- Open PEN device requirements

The question should be practical, something an installer or design engineer would need to know.
The answer should explain the rule AND the reason if available.

### OUTPUT (JSON only):
{{"question": "What are the requirements for [specific EV installation scenario]?", "answer": "[Clear procedural/design guidance] according to UKPN EDS 08-5050."}}

If the text lacks actionable design guidance, return: {{"question": "N/A", "answer": "N/A"}}
"""

TIER3_PROMPT = """
You are a Senior Grid Compliance Engineer advising commercial fleet operators on EV infrastructure.

### CONTEXT (SPEN EV Fleet Guide):
Source: {source} | Page: {page}
---
{content}
---

### TASK:
Generate a SCENARIO-BASED Question and Answer pair focused on:
- Capacity assessment (Authorized Capacity vs Demand)
- The "30% Rule" for load management
- Cost and timeline estimations
- Operational mitigation strategies (timed charging, load management)
- Fleet sizing and charging speed selection

Frame it as a real-world business decision scenario.
The answer should provide actionable guidance with any referenced percentages or timelines.

### OUTPUT (JSON only):
{{"question": "A fleet operator wants to [specific scenario]. What should they consider?", "answer": "[Practical guidance with specific figures] based on SPEN recommendations."}}

If the text lacks scenario-relevant content, return: {{"question": "N/A", "answer": "N/A"}}
"""

# Source to tier mapping
SOURCE_TIER_MAP = {
    "G99_Issue_2.pdf": ("Tier1-Fact", TIER1_PROMPT),
    "UKPN_EDS_08_5050.pdf": ("Tier2-Design", TIER2_PROMPT),
    "SPEN_EV_Fleet_Guide.pdf": ("Tier3-Scenario", TIER3_PROMPT),
}


def init_qa_table():
    """Initialize the training_dataset table with tier metadata."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    create_table_query = """
        CREATE TABLE IF NOT EXISTS training_dataset (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chunk_id INTEGER,
            source_file TEXT,
            page_number INTEGER,
            tier TEXT,
            question TEXT,
            answer TEXT,
            generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (chunk_id) REFERENCES document_chunks(id)
        )
    """
    cursor.execute(create_table_query)
    conn.commit()
    return conn


def get_merged_chunks(conn, min_chars: int = 500):
    """
    Merge consecutive chunks from the same page into larger context blocks.
    This provides better context for QA generation.
    """
    cursor = conn.cursor()
    
    # Get all chunks ordered by source and page
    cursor.execute("""
        SELECT id, source_file, page_number, content 
        FROM document_chunks 
        ORDER BY source_file, page_number, id
    """)
    
    all_chunks = cursor.fetchall()
    merged = []
    current_block = {
        "ids": [],
        "source": None,
        "page": None,
        "content": []
    }
    
    for chunk_id, source, page, content in all_chunks:
        # Check if we should merge with current block
        if (current_block["source"] == source and 
            current_block["page"] == page and 
            len("\n".join(current_block["content"])) < min_chars):
            # Add to current block
            current_block["ids"].append(chunk_id)
            current_block["content"].append(content)
        else:
            # Save current block if it has content
            if current_block["ids"]:
                merged.append({
                    "ids": current_block["ids"],
                    "source": current_block["source"],
                    "page": current_block["page"],
                    "content": "\n".join(current_block["content"])
                })
            # Start new block
            current_block = {
                "ids": [chunk_id],
                "source": source,
                "page": page,
                "content": [content]
            }
    
    # Don't forget the last block
    if current_block["ids"]:
        merged.append({
            "ids": current_block["ids"],
            "source": current_block["source"],
            "page": current_block["page"],
            "content": "\n".join(current_block["content"])
        })
    
    return merged


def get_unprocessed_chunks(conn):
    """Get chunk IDs that haven't been processed yet."""
    cursor = conn.cursor()
    cursor.execute("""
        SELECT DISTINCT chunk_id FROM training_dataset
    """)
    processed = set(row[0] for row in cursor.fetchall())
    return processed


def generate_qa_for_block(model, block: dict, processed_ids: set) -> Optional[dict]:
    """Generate a QA pair for a merged block using the appropriate tier prompt."""
    # Skip if all chunks in this block are already processed
    if all(cid in processed_ids for cid in block["ids"]):
        return None
    
    source = block["source"]
    tier_name, prompt_template = SOURCE_TIER_MAP.get(source, ("Unknown", TIER1_PROMPT))
    
    prompt = prompt_template.format(
        source=source,
        page=block["page"],
        content=block["content"]
    )
    
    try:
        response = model.generate_content(prompt)
        resp_text = response.text.strip()
        
        # Parse JSON from response
        if "```json" in resp_text:
            resp_text = resp_text.split("```json")[1].split("```")[0].strip()
        elif "```" in resp_text:
            resp_text = resp_text.split("```")[1].split("```")[0].strip()
        
        qa_pair = json.loads(resp_text)
        
        if qa_pair.get("question") != "N/A":
            return {
                "chunk_ids": block["ids"],
                "source": source,
                "page": block["page"],
                "tier": tier_name,
                "question": qa_pair["question"],
                "answer": qa_pair["answer"]
            }
    except json.JSONDecodeError as e:
        print(f"JSON parse error for block {block['ids']}: {e}")
    except Exception as e:
        print(f"Error generating QA for block {block['ids']}: {e}")
    
    return None


def generate_dataset(batch_size: int = 50, delay: float = 0.5):
    """
    Generate QA pairs using the Three-Tier strategy.
    
    Args:
        batch_size: Number of blocks to process before committing
        delay: Delay between API calls (rate limiting)
    """
    if not api_key:
        print("Error: GEMINI_API_KEY not set in environment.")
        print("Please create a .env file with: GEMINI_API_KEY=your_key_here")
        return
    
    conn = init_qa_table()
    cursor = conn.cursor()
    
    print("Merging chunks into context blocks...")
    merged_blocks = get_merged_chunks(conn, min_chars=400)
    processed_ids = get_unprocessed_chunks(conn)
    
    # Filter to unprocessed blocks
    unprocessed_blocks = [
        b for b in merged_blocks 
        if not all(cid in processed_ids for cid in b["ids"])
    ]
    
    if not unprocessed_blocks:
        print("All chunks have been processed.")
        return
    
    print(f"Total merged blocks: {len(merged_blocks)}")
    print(f"Unprocessed blocks: {len(unprocessed_blocks)}")
    print(f"Using Gemini 2.0 Flash for generation...")
    
    model = genai.GenerativeModel("gemini-2.0-flash")
    
    generated_count = 0
    skipped_count = 0
    
    for i, block in enumerate(tqdm(unprocessed_blocks, desc="Generating QA")):
        result = generate_qa_for_block(model, block, processed_ids)
        
        if result:
            # Insert into database - use first chunk_id as reference
            cursor.execute("""
                INSERT INTO training_dataset 
                (chunk_id, source_file, page_number, tier, question, answer)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                result["chunk_ids"][0],
                result["source"],
                result["page"],
                result["tier"],
                result["question"],
                result["answer"]
            ))
            generated_count += 1
            
            # Mark all chunk IDs as processed
            for cid in result["chunk_ids"]:
                processed_ids.add(cid)
        else:
            skipped_count += 1
        
        # Commit in batches
        if (i + 1) % batch_size == 0:
            conn.commit()
            print(f"\nCommitted batch {(i + 1) // batch_size}. "
                  f"Generated: {generated_count}, Skipped: {skipped_count}")
        
        # Rate limiting
        time.sleep(delay)
    
    conn.commit()
    conn.close()
    
    print(f"\n{'='*50}")
    print(f"Dataset generation complete!")
    print(f"Generated: {generated_count} QA pairs")
    print(f"Skipped: {skipped_count} blocks (N/A or already processed)")
    print(f"{'='*50}")


def show_stats():
    """Display statistics about the generated dataset."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    print("\n" + "="*60)
    print("TRAINING DATASET STATISTICS")
    print("="*60)
    
    # Total count
    cursor.execute("SELECT COUNT(*) FROM training_dataset")
    total = cursor.fetchone()[0]
    print(f"\nTotal QA pairs: {total}")
    
    # By tier
    cursor.execute("""
        SELECT tier, COUNT(*) as count 
        FROM training_dataset 
        GROUP BY tier 
        ORDER BY tier
    """)
    print("\nBy Tier:")
    for tier, count in cursor.fetchall():
        print(f"  {tier}: {count}")
    
    # By source
    cursor.execute("""
        SELECT source_file, COUNT(*) as count 
        FROM training_dataset 
        GROUP BY source_file 
        ORDER BY count DESC
    """)
    print("\nBy Source:")
    for source, count in cursor.fetchall():
        print(f"  {source}: {count}")
    
    # Sample entries
    print("\nSample QA pairs:")
    cursor.execute("""
        SELECT tier, question, substr(answer, 1, 100) as answer_preview
        FROM training_dataset 
        ORDER BY RANDOM() 
        LIMIT 3
    """)
    for tier, q, a in cursor.fetchall():
        print(f"\n  [{tier}]")
        print(f"  Q: {q}")
        print(f"  A: {a}...")
    
    conn.close()
    print("\n" + "="*60)


def export_to_jsonl(output_path: Optional[str] = None):
    """Export the training dataset to JSONL format for fine-tuning."""
    if output_path is None:
        output_path = PROJECT_ROOT / "data" / "training_data.jsonl"
    else:
        output_path = Path(output_path)
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT source_file, page_number, tier, question, answer
        FROM training_dataset
        WHERE question != 'N/A'
    """)
    
    rows = cursor.fetchall()
    
    with open(output_path, 'w') as f:
        for source, page, tier, question, answer in rows:
            entry = {
                "instruction": question,
                "input": "",
                "output": answer,
                "metadata": {
                    "source": source,
                    "page": page,
                    "tier": tier
                }
            }
            f.write(json.dumps(entry) + "\n")
    
    conn.close()
    print(f"Exported {len(rows)} QA pairs to {output_path}")
    return output_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate QA training data")
    parser.add_argument("--generate", action="store_true", help="Generate QA pairs")
    parser.add_argument("--stats", action="store_true", help="Show dataset statistics")
    parser.add_argument("--export", action="store_true", help="Export to JSONL")
    parser.add_argument("--batch-size", type=int, default=50, help="Batch size for commits")
    parser.add_argument("--delay", type=float, default=0.5, help="Delay between API calls")
    
    args = parser.parse_args()
    
    if args.generate:
        generate_dataset(batch_size=args.batch_size, delay=args.delay)
    
    if args.stats:
        show_stats()
    
    if args.export:
        export_to_jsonl()
    
    # Default: show help
    if not any([args.generate, args.stats, args.export]):
        parser.print_help()
        print("\nExample usage:")
        print("  python qa_generator.py --generate")
        print("  python qa_generator.py --stats")
        print("  python qa_generator.py --export")
