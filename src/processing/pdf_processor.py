import os
import sqlite3
from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from pypdf import PdfReader, PdfWriter


# Define project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
INTERIM_DATA_DIR = PROJECT_ROOT / "data" / "interim"
DB_PATH = PROJECT_ROOT / "data" / "pipeline.db"

# Page ranges from the plan (1-indexed based on your provided PDF list)
G99_PAGES = (
    list(range(26, 45))
    + list(range(61, 66))
    + list(range(75, 81))
    + list(range(89, 99))
    + list(range(125, 127))
    + list(range(137, 145))
    + [199]
    + list(range(312, 318))
    + list(range(318, 320))
)

UKPN_PAGES = list(range(8, 10)) + list(range(11, 15)) + [17]

SPEN_PAGES = list(range(5, 7)) + list(range(10, 16))

SLICING_CONFIG = {
    "G99_Issue_2.pdf": G99_PAGES,
    "UKPN_EDS_08_5050.pdf": UKPN_PAGES,
    "SPEN_EV_Fleet_Guide.pdf": SPEN_PAGES,
}


def init_db():
    """Initializes the SQLite database using joined strings for clean queries."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    create_table_query = "\n".join(
        [
            "CREATE TABLE IF NOT EXISTS document_chunks (",
            "    id INTEGER PRIMARY KEY AUTOINCREMENT,",
            "    source_file TEXT,",
            "    page_number INTEGER,",
            "    content TEXT,",
            "    extracted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
            ")",
        ]
    )

    cursor.execute(create_table_query)
    conn.commit()
    return conn


def slice_pdf(pdf_name, target_pages):
    """Slices specific pages from a PDF and returns the path to the interim file."""
    input_path = RAW_DATA_DIR / pdf_name
    INTERIM_DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = INTERIM_DATA_DIR / f"sliced_{pdf_name}"

    if not input_path.exists():
        return None

    reader = PdfReader(input_path)
    writer = PdfWriter()

    actual_pages_added = []
    for page_num in target_pages:
        # pypdf is 0-indexed, our config is 1-indexed
        try:
            writer.add_page(reader.pages[page_num - 1])
            actual_pages_added.append(page_num)
        except IndexError:
            continue

    with open(output_path, "wb") as f:
        writer.write(f)

    return output_path, actual_pages_added


def process_pdfs():
    """Slices PDFs first, then converts only the slices using Docling."""
    conn = init_db()
    cursor = conn.cursor()

    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True  # Enable OCR for scanned/flowchart pages
    pipeline_options.do_table_structure = True  # Enable table/flowchart extraction

    pdf_format_option = PdfFormatOption(pipeline_options=pipeline_options)
    converter = DocumentConverter(format_options={InputFormat.PDF: pdf_format_option})

    for pdf_name, target_pages in SLICING_CONFIG.items():
        print(f"--- Slicing {pdf_name} ---")
        slice_info = slice_pdf(pdf_name, target_pages)
        if not slice_info:
            print(f"Warning: {pdf_name} not found.")
            continue

        sliced_path, mapped_pages = slice_info
        print(f"Processing slice with Docling: {sliced_path}")

        result = converter.convert(sliced_path)
        doc = result.document

        # Group content by page number
        pages_content = {}
        for element, _level in doc.iterate_items():
            if element.prov:
                # Docling page_no is 1-indexed for the sliced file
                sliced_page_no = element.prov[0].page_no
                
                # Initialize page content list if not exists
                if sliced_page_no not in pages_content:
                    pages_content[sliced_page_no] = []
                
                # Get element content as markdown
                try:
                    if hasattr(element, "export_to_markdown"):
                        content = element.export_to_markdown(doc)
                    elif hasattr(element, "text"):
                        content = element.text
                    else:
                        content = str(element)
                except Exception:
                    content = str(element)
                
                if content and content.strip():
                    pages_content[sliced_page_no].append(content.strip())

        # Remove unwanted lines from content
        for sliced_page_no in pages_content:
            pages_content[sliced_page_no] = [
                line for line in pages_content[sliced_page_no] 
                if "<!-- 🖼️❌ Image not available. Please use `PdfPipelineOptions(generate_picture_images=True)` -->" not in line
            ]

        # Now insert one entry per page with all content combined
        for sliced_page_no, content_list in sorted(pages_content.items()):
            # Map back to original page number
            original_page_no = mapped_pages[sliced_page_no - 1]
            
            # Combine all content from this page into markdown
            page_markdown = "\n\n".join(content_list)
            
            insert_query = "\n".join([
                "INSERT INTO document_chunks (source_file, page_number, content)",
                "VALUES (?, ?, ?)",
            ])
            cursor.execute(insert_query, (pdf_name, original_page_no, page_markdown))

        conn.commit()
        print(f"Finished processing and mapping {pdf_name}")

    conn.close()


if __name__ == "__main__":

    process_pdfs()
