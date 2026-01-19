import json
import os
import sqlite3
import time
from pathlib import Path

import google.generativeai as genai
from dotenv import load_dotenv
from tqdm import tqdm

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DB_PATH = PROJECT_ROOT / "data" / "pipeline.db"

GEN_PROMPT = """
You are a Senior Grid Compliance Engineer. Below is a technical snippet from an energy regulation document.
Your task is to generate a high-quality Question and Answer pair based strictly on this text.

### CONTEXT:
Source Document: {source}
Page Number: {page}
Text Content: 
{content}

### INSTRUCTIONS:
1. Generate one highly technical Question.
2. Generate a precise and legally accurate Answer based on the provided text.
3. Format the output as a JSON object with keys "question" and "answer".
4. If the text does not contain enough information to generate a meaningful question, return {"question": "N/A", "answer": "N/A"}.

### OUTPUT SCHEMA (JSON ONLY):
{
  "question": "...",
  "answer": "..."
}
"""


def init_qa_table():
    """Adds the training_dataset table using clean joined SQL strings."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    create_table_query = "\n".join(
        [
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
    )

    cursor.execute(create_table_query)
    conn.commit()
    return conn


def generate_dataset():
    """Generates QA pairs using clean SQL queries."""
    conn = init_qa_table()
    cursor = conn.cursor()

    select_query = "\n".join(
        [
            "SELECT id, source_file, page_number, content ",
            "FROM document_chunks ",
            "WHERE id NOT IN (SELECT chunk_id FROM training_dataset)",
        ]
    )

    cursor.execute(select_query)
    chunks = cursor.fetchall()

    if not chunks:
        print("No new chunks to process.")
        return

    print(f"Generating QA for {len(chunks)} chunks using Gemini 2.0 Flash...")
    model = genai.GenerativeModel("gemini-2.0-flash")

    for chunk_id, source, page, content in tqdm(chunks):
        prompt = GEN_PROMPT.format(source=source, page=page, content=content)

        try:
            response = model.generate_content(prompt)
            resp_text = response.text.strip()
            if "```json" in resp_text:
                resp_text = resp_text.split("```json")[1].split("```")[0].strip()
            elif "```" in resp_text:
                resp_text = resp_text.split("```")[1].split("```")[0].strip()

            qa_pair = json.loads(resp_text)

            if qa_pair.get("question") != "N/A":
                insert_query = "\n".join(
                    [
                        "INSERT INTO training_dataset (chunk_id, source_file, page_number, question, answer)",
                        "VALUES (?, ?, ?, ?, ?)",
                    ]
                )
                cursor.execute(
                    insert_query,
                    (chunk_id, source, page, qa_pair["question"], qa_pair["answer"]),
                )

            conn.commit()
            time.sleep(0.5)

        except Exception as e:
            print(f"Error generating QA for chunk {chunk_id}: {e}")
            continue

    conn.close()
    print("Dataset generation complete.")


if __name__ == "__main__":
    generate_dataset()
