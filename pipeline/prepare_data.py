"""
pipeline/prepare_data.py

Consolidated script for diabetes RAG data preparation:
  1. Fetch abstracts from PubMed NCBI API
  2. Chunk abstracts (fixed-size or recursive)
  3. Write chunks to data/chunks_{strategy}/diabetes.jsonl

Output format is compatible with build_bm25.py and embed_and_upsert.py

Usage:
    python pipeline/prepare_data.py --strategy fixed
    python pipeline/prepare_data.py --strategy recursive

Requirements:
    pip install requests pyyaml rank_bm25
"""

import argparse
import json
import logging
import re
import requests
import time
from pathlib import Path
from xml.etree import ElementTree as ET
from dataclasses import asdict, dataclass

import yaml

# Configuration
DATA_DIR = Path("data/repos")
SEARCH_QUERIES = [
    # Broad disease terms
    "type 2 diabetes mellitus",
    "T2DM management",
    "non-insulin dependent diabetes",
    "adult onset diabetes",

    # Pathophysiology
    "insulin resistance mechanism",
    "beta cell dysfunction diabetes",
    "pancreatic insulin secretion",
    "glucose intolerance pathophysiology",
    "hyperglycemia type 2 diabetes",

    # Treatment / medications
    "metformin type 2 diabetes",
    "SGLT2 inhibitors diabetes treatment",
    "GLP-1 receptor agonist diabetes",
    "DPP-4 inhibitors diabetes",
    "insulin therapy type 2 diabetes",
    "sulfonylurea diabetes treatment",

    # Complications
    "diabetic nephropathy type 2",
    "diabetic retinopathy",
    "diabetic neuropathy",
    "cardiovascular disease diabetes",
    "diabetes foot complications",

    # Lifestyle / prevention
    "diabetes diet intervention",
    "physical activity type 2 diabetes",
    "weight loss diabetes remission",
    "bariatric surgery diabetes",
    "Mediterranean diet diabetes",

    # Comorbidities
    "obesity type 2 diabetes",
    "diabetes hypertension",
    "diabetes dyslipidemia",
    "metabolic syndrome diabetes",
    "fatty liver diabetes",

    # Diagnosis / monitoring
    "HbA1c diabetes monitoring",
    "blood glucose self monitoring diabetes",
    "continuous glucose monitoring diabetes",
    "diabetes screening diagnosis",
    "fasting glucose diabetes diagnosis",

    # Special populations
    "diabetes elderly patients",
    "gestational diabetes type 2 risk",
    "diabetes pregnancy",
    "diabetes in South Asia",
    "childhood obesity diabetes risk",
]
RESULTS_PER_QUERY = 80
BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
EMAIL = "your_email@example.com"

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# Data model
@dataclass
class Chunk:
    chunk_id: str
    title: str
    abstract: str
    authors: str
    year: str
    journal: str
    chunk_strategy: str


# PubMed fetching
def search_pubmed(query: str, max_results: int = 80, retries: int = 3) -> list[str]:
    """Search PubMed and return list of PMIDs."""
    params = {
        "db": "pubmed",
        "term": query,
        "retmax": max_results,
        "retmode": "json",
        "email": EMAIL,
        "sort": "relevance",
    }
    for attempt in range(retries):
        try:
            resp = requests.get(BASE_URL + "esearch.fcgi", params=params, timeout=15)
            resp.raise_for_status()
            return resp.json()["esearchresult"]["idlist"]
        except Exception as e:
            log.debug("Search attempt %d failed: %s", attempt + 1, e)
            time.sleep(2)
    return []


def fetch_abstracts(pmid_list: list[str], retries: int = 3) -> str:
    """Fetch XML for a list of PMIDs from PubMed."""
    ids = ",".join(pmid_list)
    params = {
        "db": "pubmed",
        "id": ids,
        "retmode": "xml",
        "rettype": "abstract",
        "email": EMAIL,
    }
    for attempt in range(retries):
        try:
            resp = requests.get(BASE_URL + "efetch.fcgi", params=params, timeout=20)
            resp.raise_for_status()
            return resp.text
        except Exception as e:
            log.debug("Fetch attempt %d failed: %s", attempt + 1, e)
            time.sleep(3)
    return ""


def parse_xml(xml_text: str) -> list[dict]:
    """Parse PubMed XML and extract structured abstract records."""
    if not xml_text:
        return []

    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as e:
        log.warning("XML parse error: %s", e)
        return []

    records = []
    for article in root.findall(".//PubmedArticle"):
        try:
            pmid = article.findtext(".//PMID", default="").strip()

            title_el = article.find(".//ArticleTitle")
            title = "".join(title_el.itertext()).strip() if title_el is not None else ""

            abstract_parts = article.findall(".//AbstractText")
            if not abstract_parts:
                continue

            abstract_text = " ".join(
                (part.get("Label", "") + ": " if part.get("Label") else "") + (part.text or "")
                for part in abstract_parts
            ).strip()

            # Skip very short abstracts
            if len(abstract_text.split()) < 40:
                continue

            year = article.findtext(".//PubDate/Year", default="")
            if not year:
                medline_date = article.findtext(".//PubDate/MedlineDate", default="")
                year = medline_date[:4] if medline_date else ""

            journal = article.findtext(".//Journal/Title", default="")

            authors = []
            for author in article.findall(".//Author"):
                last = author.findtext("LastName", default="")
                first = author.findtext("ForeName", default="")
                if last:
                    authors.append(f"{last} {first}".strip())
            author_str = ", ".join(authors[:3])
            if len(authors) > 3:
                author_str += " et al."

            records.append({
                "pmid": pmid,
                "title": title,
                "abstract": abstract_text,
                "year": year,
                "journal": journal,
                "authors": author_str,
            })

        except Exception:
            continue

    return records


def fetch_all_abstracts() -> list[dict]:
    """Fetch diabetes abstracts from PubMed using all search queries."""
    all_records = []
    seen_pmids = set()

    log.info("=" * 60)
    log.info("Diabetes RAG - PubMed Abstract Fetcher")
    log.info("Target: 550+ abstracts")
    log.info("=" * 60)

    for i, query in enumerate(SEARCH_QUERIES, 1):
        log.info("[%d/%d] %s", i, len(SEARCH_QUERIES), query)

        pmids = search_pubmed(query, max_results=RESULTS_PER_QUERY)
        new_pmids = [p for p in pmids if p not in seen_pmids]
        log.info("  %d found - %d new", len(pmids), len(new_pmids))

        if not new_pmids:
            continue

        # Fetch in batches of 20
        for start in range(0, len(new_pmids), 20):
            batch = new_pmids[start:start + 20]
            xml_text = fetch_abstracts(batch)
            records = parse_xml(xml_text)
            all_records.extend(records)
            seen_pmids.update(batch)
            log.info("  Batch %d: +%d records | Total: %d",
                     start // 20 + 1, len(records), len(all_records))

            # Stop early if target reached
            if len(all_records) >= 600:
                log.info("Target reached (%d abstracts). Stopping.", len(all_records))
                return all_records

            time.sleep(0.4)

        if len(all_records) >= 600:
            return all_records

    return all_records


# Chunking strategies
def chunk_fixed(text: str, chunk_size: int, chunk_overlap: int) -> list[tuple[int, int, str]]:
    """
    Split text into fixed-token-count windows.
    Returns list of (char_start, char_end, text).
    """
    words = text.split()
    words_per_chunk = max(50, chunk_size // 4)
    overlap_words = max(5, chunk_overlap // 4)

    chunks = []
    i = 0
    while i < len(words):
        end = min(i + words_per_chunk, len(words))
        chunk_text = " ".join(words[i:end])
        if chunk_text.strip():
            char_start = len(" ".join(words[:i]))
            char_end = len(" ".join(words[:end]))
            chunks.append((char_start, char_end, chunk_text))
        i += words_per_chunk - overlap_words

    return chunks


def chunk_recursive(text: str, chunk_size: int, chunk_overlap: int, separators: list[str]) -> list[tuple[int, int, str]]:
    """
    Split text on sentence boundaries, then by size.
    Returns list of (char_start, char_end, text).
    """
    # Split by sentences first
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    chunks = []
    current_chunk = []
    current_size = 0
    char_start = 0

    for sentence in sentences:
        sentence_size = len(sentence.split())
        
        if current_size + sentence_size > chunk_size and current_chunk:
            chunk_text = " ".join(current_chunk)
            char_end = char_start + len(chunk_text)
            chunks.append((char_start, char_end, chunk_text))
            
            # Overlap: keep last half of chunk
            overlap_count = max(1, len(current_chunk) // 2)
            current_chunk = current_chunk[-overlap_count:]
            current_size = sum(len(s.split()) for s in current_chunk)
            char_start = char_end - len(" ".join(current_chunk))
        
        current_chunk.append(sentence)
        current_size += sentence_size

    if current_chunk:
        chunk_text = " ".join(current_chunk)
        char_end = char_start + len(chunk_text)
        chunks.append((char_start, char_end, chunk_text))

    return chunks


def create_chunks(abstracts: list[dict], strategy: str, cfg: dict) -> list[Chunk]:
    """
    Convert abstracts into chunks using specified strategy.
    """
    chunks = []
    chunking_cfg = cfg["chunking"][strategy]
    chunk_size = chunking_cfg.get("chunk_size", 512)
    chunk_overlap = chunking_cfg.get("chunk_overlap", 64)
    separators = chunking_cfg.get("separators", ["\n", " "])

    for abstract in abstracts:
        pmid = abstract["pmid"]
        title = abstract["title"]
        abstract_text = abstract["abstract"]
        authors = abstract["authors"]
        year = abstract["year"]
        journal = abstract["journal"]

        # Chunk the abstract text
        if strategy == "fixed":
            raw_chunks = chunk_fixed(abstract_text, chunk_size, chunk_overlap)
        elif strategy == "recursive":
            raw_chunks = chunk_recursive(abstract_text, chunk_size, chunk_overlap, separators)
        else:
            raw_chunks = [(0, len(abstract_text), abstract_text)]

        # Create Chunk objects
        for idx, (char_start, char_end, chunk_text) in enumerate(raw_chunks):
            if not chunk_text.strip():
                continue

            chunk_id = f"{pmid}::{idx}"

            chunk = Chunk(
                chunk_id=chunk_id,
                title=title,
                abstract=chunk_text,
                authors=authors,
                year=year,
                journal=journal,
                chunk_strategy=strategy,
            )
            chunks.append(chunk)

    return chunks


def save_chunks(chunks: list[Chunk], strategy: str) -> None:
    """Save chunks to JSONL format."""
    output_dir = Path(f"data/chunks_{strategy}")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "diabetes.jsonl"

    with open(output_path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(asdict(chunk)) + "\n")

    log.info("Saved %d chunks to %s", len(chunks), output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Fetch and prepare diabetes abstracts for RAG pipeline"
    )
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["fixed", "recursive"],
        default="fixed",
        help="Chunking strategy (default: fixed)",
    )
    args = parser.parse_args()

    log.info("Loading config...")
    cfg = yaml.safe_load(open("config.yaml"))

    # Fetch abstracts
    log.info("")
    abstracts = fetch_all_abstracts()

    log.info("")
    log.info("=" * 60)
    log.info("Final count: %d abstracts", len(abstracts))
    log.info("=" * 60)

    # Create chunks
    log.info("")
    log.info("Creating %s chunks...", args.strategy)
    chunks = create_chunks(abstracts, args.strategy, cfg)
    log.info("Total chunks: %d", len(chunks))

    # Save chunks
    save_chunks(chunks, args.strategy)
    log.info("")
    log.info("Data ready for: python pipeline/build_bm25.py --strategy %s", args.strategy)


if __name__ == "__main__":
    main()
