"""
PubMed Abstract Fetcher — Target: 550+ abstracts
=================================================
Fixes for low count:
  1. More queries (40 instead of 10)
  2. Higher per-query limit (80 instead of 60)
  3. Broader search terms (not just "type 2 diabetes X")
  4. MeSH term queries for better PubMed coverage
  5. Retries on failed batches

Output:
  diabetes_abstracts.json
  diabetes_abstracts.csv
  diabetes_corpus.txt

Usage:
  python pubmed_fetcher.py

Requirements:
  pip install requests
"""

import requests
import json
import csv
import time
from xml.etree import ElementTree as ET

EMAIL = "your_email@example.com"   # ← change this to your email (required by NCBI)

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

RESULTS_PER_QUERY = 80     # 40 queries × 80 = 3200 attempts → ~550+ unique after dedup
OUTPUT_JSON = "diabetes_abstracts.json"
OUTPUT_CSV  = "diabetes_abstracts.csv"
OUTPUT_TXT  = "diabetes_corpus.txt"
BASE_URL    = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"


def search_pubmed(query, max_results=80, retries=3):
    """Search PubMed and return list of PMIDs."""
    params = {
        "db":      "pubmed",
        "term":    query,
        "retmax":  max_results,
        "retmode": "json",
        "email":   EMAIL,
        "sort":    "relevance",
    }
    for attempt in range(retries):
        try:
            resp = requests.get(BASE_URL + "esearch.fcgi", params=params, timeout=15)
            resp.raise_for_status()
            return resp.json()["esearchresult"]["idlist"]
        except Exception as e:
            print(f"    Search attempt {attempt+1} failed: {e}")
            time.sleep(2)
    return []


def fetch_abstracts(pmid_list, retries=3):
    """Fetch XML for a list of PMIDs."""
    ids = ",".join(pmid_list)
    params = {
        "db":      "pubmed",
        "id":      ids,
        "retmode": "xml",
        "rettype": "abstract",
        "email":   EMAIL,
    }
    for attempt in range(retries):
        try:
            resp = requests.get(BASE_URL + "efetch.fcgi", params=params, timeout=20)
            resp.raise_for_status()
            return resp.text
        except Exception as e:
            print(f"    Fetch attempt {attempt+1} failed: {e}")
            time.sleep(3)
    return ""


def parse_xml(xml_text):
    """Parse PubMed XML and extract structured records."""
    if not xml_text:
        return []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as e:
        print(f"    XML parse error: {e}")
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

            # Skip very short abstracts — not useful for RAG
            if len(abstract_text.split()) < 40:
                continue

            year = article.findtext(".//PubDate/Year", default="")
            if not year:
                medline_date = article.findtext(".//PubDate/MedlineDate", default="")
                year = medline_date[:4] if medline_date else ""

            journal = article.findtext(".//Journal/Title", default="")

            authors = []
            for author in article.findall(".//Author"):
                last  = author.findtext("LastName", default="")
                first = author.findtext("ForeName", default="")
                if last:
                    authors.append(f"{last} {first}".strip())
            author_str = ", ".join(authors[:3])
            if len(authors) > 3:
                author_str += " et al."

            document = (
                f"Title: {title}\n"
                f"Authors: {author_str}\n"
                f"Year: {year}\n"
                f"Journal: {journal}\n\n"
                f"Abstract: {abstract_text}"
            )

            records.append({
                "pmid":       pmid,
                "title":      title,
                "abstract":   abstract_text,
                "year":       year,
                "journal":    journal,
                "authors":    author_str,
                "document":   document,
                "word_count": len(abstract_text.split()),
            })

        except Exception:
            continue

    return records


def main():
    all_records = []
    seen_pmids  = set()

    print("=" * 60)
    print("  PubMed Diabetes Abstract Fetcher — Target: 550+")
    print("=" * 60)

    for i, query in enumerate(SEARCH_QUERIES, 1):
        print(f"\n[{i}/{len(SEARCH_QUERIES)}] '{query}'")

        pmids     = search_pubmed(query, max_results=RESULTS_PER_QUERY)
        new_pmids = [p for p in pmids if p not in seen_pmids]
        print(f"  {len(pmids)} found → {len(new_pmids)} new")

        if not new_pmids:
            continue

        # Fetch in batches of 20
        for start in range(0, len(new_pmids), 20):
            batch    = new_pmids[start:start + 20]
            xml_text = fetch_abstracts(batch)
            records  = parse_xml(xml_text)
            all_records.extend(records)
            seen_pmids.update(batch)
            print(f"  Batch {start//20 + 1}: +{len(records)} | Total: {len(all_records)}")

            # Stop early if we've hit the target
            if len(all_records) >= 600:
                print(f"\n  Target reached ({len(all_records)} abstracts). Stopping early.")
                break

            time.sleep(0.4)   # max 3 req/sec without API key

        if len(all_records) >= 600:
            break

    # ── Save outputs ───────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Final count: {len(all_records)} abstracts")
    print(f"{'='*60}")

    # JSON (used by Kaggle notebook)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(all_records, f, indent=2, ensure_ascii=False)
    print(f"[✓] {OUTPUT_JSON}")

    # CSV (for reference / reporting)
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["pmid","title","authors","year","journal","abstract","word_count"])
        writer.writeheader()
        for r in all_records:
            writer.writerow({k: r[k] for k in ["pmid","title","authors","year","journal","abstract","word_count"]})
    print(f"[✓] {OUTPUT_CSV}")

    # Plain text corpus (for inspection)
    with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
        for r in all_records:
            f.write(r["document"] + "\n\n---\n\n")
    print(f"[✓] {OUTPUT_TXT}")

    # Stats
    total_words = sum(r["word_count"] for r in all_records)
    avg_words   = total_words // max(len(all_records), 1)
    print(f"\n  Avg abstract length : {avg_words} words")
    print(f"  Total words in corpus: {total_words:,}")
    print(f"\n  Sample preview:")
    print("-" * 55)
    if all_records:
        print(all_records[0]["document"][:500] + "...")


if __name__ == "__main__":
    main()
