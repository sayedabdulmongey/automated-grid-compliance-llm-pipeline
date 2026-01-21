"""
QA Generator - Three-Tier Strategy for Grid Compliance Training Data

Tier 1 (Fact): Hard numerical limits from G99 tables
Tier 2 (Design): Compliance rules from UKPN EDS
Tier 3 (Scenario): Fleet logistics and project estimation from SPEN Guide

Usage:
    python qa_generator.py --generate
    python qa_generator.py --stats
    python qa_generator.py --push-hf
"""

import json
import os
import sqlite3
import time
from pathlib import Path

from datasets import Dataset
from dotenv import load_dotenv
from groq import Groq
from huggingface_hub import login
from pydantic import BaseModel
from tqdm import tqdm

# Load environment variables
load_dotenv()

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DB_PATH = PROJECT_ROOT / "data" / "pipeline.db"
OUTPUT_DIR = PROJECT_ROOT / "data" / "output"

# Configure Groq
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")
HF_REPO_ID = os.getenv("HF_REPO_ID")

if GROQ_API_KEY:
    client = Groq(api_key=GROQ_API_KEY)
else:
    client = None

# Groq configuration
MODEL_NAME = "openai/gpt-oss-120b"

# Rate limiting config
REQUESTS_BEFORE_SLEEP = 20
SLEEP_DURATION = 120


# =============================================================================
# Pydantic Schema for Structured Output
# =============================================================================


class QAPair(BaseModel):
    question: str
    answer: str


class QAList(BaseModel):
    items: list[QAPair]


# =============================================================================
# Three-Tier Prompt Templates
# =============================================================================

TIER1_PROMPT_LINES = [
    "You are a Senior Grid Compliance Engineer specializing in UK power grid regulations.",
    "",
    "### CONTEXT (G99 Engineering Recommendation):",
    "Source: {source} | Page: {page}",
    "---",
    "{content}",
    "---",
    "",
    "### TASK:",
    "Generate as many diverse FACTUAL Question and Answer pairs as possible from this page.",
    "Focus on:",
    "- Specific numerical thresholds (kW, kVA, Hz, voltage levels)",
    "- Equipment classifications (Type A, B, C, D)",
    "- Protection settings and timing requirements",
    "- Frequency response parameters",
    "",
    "Each question should be something a junior engineer would ask when checking compliance.",
    "Each answer must cite the specific numerical value or classification from the text.",
    "",
    "If the text lacks specific technical data, return an empty list.",
    "",
    "Return a JSON object with this structure:",
    '{{"items": [{{"question": "...", "answer": "..."}}, ...]}}',
]

TIER2_PROMPT_LINES = [
    "You are a Senior Grid Compliance Engineer specializing in UK Distribution Network Operator (DNO) requirements.",
    "",
    "### CONTEXT (UKPN EDS 08-5050 - EV Connections):",
    "Source: {source} | Page: {page}",
    "---",
    "{content}",
    "---",
    "",
    "### TASK:",
    "Generate as many diverse DESIGN/PROCEDURAL Question and Answer pairs as possible from this page.",
    "Focus on:",
    "- Connection application processes",
    "- Earthing and safety requirements (PME, TT systems)",
    "- Physical separation rules (2.5m requirement, etc.)",
    "- Metering and notification procedures",
    "- Open PEN device requirements",
    "",
    "Each question should be practical, something an installer or design engineer would need to know.",
    "Each answer should explain the rule AND the reason if available.",
    "",
    "If the text lacks actionable design guidance, return an empty list.",
    "",
    "Return a JSON object with this structure:",
    '{{"items": [{{"question": "...", "answer": "..."}}, ...]}}',
]

TIER3_PROMPT_LINES = [
    "You are a Senior Grid Compliance Engineer advising commercial fleet operators on EV infrastructure.",
    "",
    "### CONTEXT (SPEN EV Fleet Guide):",
    "Source: {source} | Page: {page}",
    "---",
    "{content}",
    "---",
    "",
    "### TASK:",
    "Generate as many diverse SCENARIO-BASED Question and Answer pairs as possible from this page.",
    "Focus on:",
    "- Capacity assessment (Authorized Capacity vs Demand)",
    '- The "30% Rule" for load management',
    "- Cost and timeline estimations",
    "- Operational mitigation strategies (timed charging, load management)",
    "- Fleet sizing and charging speed selection",
    "",
    "Frame each as a real-world business decision scenario.",
    "Each answer should provide actionable guidance with any referenced percentages or timelines.",
    "",
    "If the text lacks scenario-relevant content, return an empty list.",
    "",
    "Return a JSON object with this structure:",
    '{{"items": [{{"question": "...", "answer": "..."}}, ...]}}',
]

TIER1_PROMPT = "\n".join(TIER1_PROMPT_LINES)
TIER2_PROMPT = "\n".join(TIER2_PROMPT_LINES)
TIER3_PROMPT = "\n".join(TIER3_PROMPT_LINES)

# Source to prompt mapping
SOURCE_PROMPT_MAP = {
    "G99_Issue_2.pdf": TIER1_PROMPT,
    "UKPN_EDS_08_5050.pdf": TIER2_PROMPT,
    "SPEN_EV_Fleet_Guide.pdf": TIER3_PROMPT,
}


def init_qa_table():
    """Initialize the training_dataset table."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    create_lines = [
        "CREATE TABLE IF NOT EXISTS training_dataset (",
        "    id INTEGER PRIMARY KEY AUTOINCREMENT,",
        "    chunk_id INTEGER,",
        "    source_file TEXT,",
        "    page_number INTEGER,",
        "    question TEXT,",
        "    answer TEXT,",
        "    generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,",
        "    FOREIGN KEY (chunk_id) REFERENCES document_chunks(id)",
        ")",
    ]
    cursor.execute("\n".join(create_lines))
    conn.commit()
    return conn


def get_all_chunks(conn):
    """Get all chunks from the database."""
    cursor = conn.cursor()
    select_lines = [
        "SELECT id, source_file, page_number, content",
        "FROM document_chunks",
        "ORDER BY source_file, page_number",
    ]
    cursor.execute("\n".join(select_lines))
    return cursor.fetchall()


def get_processed_chunk_ids(conn):
    """Get chunk IDs that have already been processed."""
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT chunk_id FROM training_dataset")
    return set(row[0] for row in cursor.fetchall())


def generate_qa_for_chunk(chunk_id, source, page, content):
    """Generate multiple QA pairs for a single chunk using Pydantic schema."""
    prompt_template = SOURCE_PROMPT_MAP.get(source, TIER1_PROMPT)
    prompt = prompt_template.format(source=source, page=page, content=content)

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": "You are a technical documentation expert. Always respond with valid JSON.",
                },
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.7,
        )

        response_text = response.choices[0].message.content
        qa_list = QAList.model_validate_json(response_text)

        results = []
        for qa in qa_list.items:
            if qa.question and qa.answer:
                results.append(
                    {
                        "chunk_id": chunk_id,
                        "source": source,
                        "page": page,
                        "question": qa.question,
                        "answer": qa.answer,
                    }
                )
        return results

    except Exception as e:
        print(f"Error generating QA for chunk {chunk_id}: {e}")

    return []


def generate_dataset():
    """Generate QA pairs from all chunks in the database."""
    if not GROQ_API_KEY:
        print("Error: GROQ_API_KEY not set in .env file")
        return

    if not client:
        print("Error: Failed to initialize Groq client")
        return

    conn = init_qa_table()

    print("Loading chunks from database...")
    all_chunks = get_all_chunks(conn)
    processed_ids = get_processed_chunk_ids(conn)

    unprocessed = [c for c in all_chunks if c[0] not in processed_ids]

    if not unprocessed:
        print("All chunks have been processed.")
        conn.close()
        return

    print(f"Total chunks: {len(all_chunks)}")
    print(f"Already processed: {len(processed_ids)}")
    print(f"To process: {len(unprocessed)}")
    print(f"Using Groq ({MODEL_NAME}) with Pydantic schema...")
    print(f"Sleep every {REQUESTS_BEFORE_SLEEP} requests for {SLEEP_DURATION}s")

    cursor = conn.cursor()

    total_generated = 0
    request_count = 0

    for chunk_id, source, page, content in tqdm(unprocessed, desc="Generating QA"):
        qa_results = generate_qa_for_chunk(chunk_id, source, page, content)

        # Only save if we got results
        if qa_results:
            for qa in qa_results:
                insert_lines = [
                    "INSERT INTO training_dataset",
                    "(chunk_id, source_file, page_number, question, answer)",
                    "VALUES (?, ?, ?, ?, ?)",
                ]
                cursor.execute(
                    "\n".join(insert_lines),
                    (
                        qa["chunk_id"],
                        qa["source"],
                        qa["page"],
                        qa["question"],
                        qa["answer"],
                    ),
                )
                total_generated += 1

            conn.commit()

        request_count += 1

        # Rate limiting
        if request_count % REQUESTS_BEFORE_SLEEP == 0:
            print(f"\nSleeping {SLEEP_DURATION}s to avoid rate limits...")
            time.sleep(SLEEP_DURATION)

    conn.close()
    print(f"\nGeneration complete! Total QA pairs created: {total_generated}")


def show_stats():
    """Display statistics about the generated dataset."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    print("\n" + "=" * 60)
    print("TRAINING DATASET STATISTICS")
    print("=" * 60)

    cursor.execute("SELECT COUNT(*) FROM training_dataset")
    total = cursor.fetchone()[0]
    print(f"\nTotal QA pairs: {total}")

    cursor.execute(
        "SELECT source_file, COUNT(*) FROM training_dataset GROUP BY source_file"
    )
    print("\nBy Source:")
    for source, count in cursor.fetchall():
        print(f"  {source}: {count}")

    print("\nSample QA pairs:")
    cursor.execute(
        "SELECT source_file, question, answer FROM training_dataset ORDER BY RANDOM() LIMIT 3"
    )
    for source, q, a in cursor.fetchall():
        print(f"\n  [{source}]")
        print(f"  Q: {q}")
        print(f"  A: {a[:150]}...")

    conn.close()
    print("\n" + "=" * 60)


def export_to_jsonl():
    """Export the training dataset to JSONL format."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "training_data.jsonl"

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    select_lines = [
        "SELECT source_file, page_number, question, answer",
        "FROM training_dataset",
    ]
    cursor.execute("\n".join(select_lines))
    rows = cursor.fetchall()

    with open(output_path, "w", encoding="utf-8") as f:
        for source, page, question, answer in rows:
            entry = {
                "input": question,
                "output": answer,
                "metadata": {
                    "source": source,
                    "page": page,
                },
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    conn.close()
    print(f"Exported {len(rows)} QA pairs to {output_path}")
    return output_path


def push_to_huggingface():
    """Push the dataset to HuggingFace Hub."""
    if not HF_TOKEN:
        print("Error: HF_TOKEN not set in .env file")
        return

    if not HF_REPO_ID:
        print("Error: HF_REPO_ID not set in .env file")
        return

    login(token=HF_TOKEN)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    select_lines = [
        "SELECT source_file, page_number, question, answer",
        "FROM training_dataset",
    ]
    cursor.execute("\n".join(select_lines))
    rows = cursor.fetchall()
    conn.close()

    data = {
        "input": [],
        "output": [],
        "source": [],
        "page": [],
    }

    for source, page, question, answer in rows:
        data["input"].append(question)
        data["output"].append(answer)
        data["source"].append(source)
        data["page"].append(page)

    dataset = Dataset.from_dict(data)
    dataset.push_to_hub(HF_REPO_ID, private=False)
    print(f"Pushed {len(rows)} QA pairs to HuggingFace: {HF_REPO_ID}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate QA training data")
    parser.add_argument("--generate", action="store_true", help="Generate QA pairs")
    parser.add_argument("--stats", action="store_true", help="Show dataset statistics")
    parser.add_argument("--export", action="store_true", help="Export to JSONL")
    parser.add_argument(
        "--push-hf", action="store_true", help="Push to HuggingFace Hub"
    )

    args = parser.parse_args()

    if args.generate:
        generate_dataset()

    if args.stats:
        show_stats()

    if args.export:
        export_to_jsonl()

    if args.push_hf:
        export_to_jsonl()
        push_to_huggingface()

    if not any([args.generate, args.stats, args.export, args.push_hf]):
        parser.print_help()
        print("\nExamples:")
        print("  python qa_generator.py --generate")
        print("  python qa_generator.py --stats")
        print("  python qa_generator.py --export")
        print("  python qa_generator.py --push-hf")
