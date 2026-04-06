# Type 2 Diabetes RAG 🩺

## About

Type 2 Diabetes RAG is a medical research synthesis system that leverages PubMed abstracts to provide evidence-based answers about Type 2 Diabetes, treatments, and related health topics. It indexes over 600 peer-reviewed abstracts from biomedical literature, enabling sophisticated queries with inline academic citations.

## System Overview

The platform uses advanced retrieval-augmented generation techniques:

- **Hybrid retrieval**: BM25 keyword search combined with semantic embeddings (all-MiniLM-L6-v2)
- **Cross-encoder reranking**: State-of-the-art models refine search results for medical relevance
- **Academic citations**: Inline `Authors (Year)` format with clickable references
- **PubMed integration**: Direct access to peer-reviewed diabetes research

## Data Sources

The system indexes peer-reviewed research across:

- **Type 2 Diabetes pathophysiology**: Insulin resistance, beta cell dysfunction
- **Pharmacological treatments**: Metformin, GLP-1 agonists, SGLT2 inhibitors, sulfonylureas
- **Lifestyle interventions**: Diet, exercise, weight management
- **Complications management**: Cardiovascular disease, kidney disease, neuropathy
- **Clinical outcomes**: HbA1c control, cardiovascular protection

Total: 600+ abstracts, 1,300+ evidence-rich chunks indexed.

## Use Cases

This system helps you understand:

- Clinical evidence for diabetes treatment options
- Mechanisms of action for major drug classes
- Lifestyle intervention effectiveness
- Risk factor relationships and complication prevention
- Latest research findings with proper attribution

## Citation Standards

All responses strictly follow academic citation format:
- **Required format**: `Authors (Year)` for every claim
- **Source**: Only PubMed abstracts provided
- **Honesty**: No invented citations or unsupported claims
- **Clickability**: All citations are interactive links to original sources

## Medical Disclaimer

This tool provides information from peer-reviewed literature and is not a substitute for professional medical advice. Always consult with qualified healthcare providers for diagnosis and treatment decisions.
