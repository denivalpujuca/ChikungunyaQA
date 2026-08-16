# ChikungunyaQA: A Clinical Question-Answering Dataset

[![Zenodo DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20444766.svg)](https://doi.org/10.5281/zenodo.20444766)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)

This repository contains the official dataset, clinical source guidelines, and end-to-end reproduction pipeline for **ChikungunyaQA**, presented in the manuscript:

> **ChikungunyaQA: A Clinical Dataset of Questions and Answers on Chikungunya**  
> *Data (MDPI), 2026.*

---

## Repository Overview

ChikungunyaQA is a multi-persona question and answer (QA) dataset in Brazilian Portuguese (PT-BR), derived from official clinical management guidelines published by the Brazilian Ministry of Health and complementary regulatory documents. 

### Key Dataset Features
- **1,262 High-Fidelity QA Pairs:** Fully grounded in official Brazilian Unified Health System (SUS) protocols.
- **Multi-Persona Architecture:**
  - **Physician (Médica):** 556 pairs (44.1%) — formal medical terminology, dosage guidelines, and clinical workflows.
  - **Patient (Paciente):** 581 pairs (46.0%) — layperson language, accessible terms, and general symptom identification.
  - **Caregiver (Cuidador):** 125 pairs (9.9%) — practical home care, monitoring warning signs, and patient support.
- **Dual Validation:**
  - Automated evaluation via an independent LLM judge (**Claude Sonnet 4.5**, G-Eval mean score: 97.1/100).
  - Blinded human evaluation by healthcare professionals (**Physician** and **Physical Therapist**).

---

## Repository Structure

```
.
├── data/                       # ChikungunyaQA dataset files
│   ├── ChikungunyaQA.jsonl     # Full research format (JSONL) with complete metadata
│   ├── ChikungunyaQA.csv       # Full research format (CSV)
│   └── ChikungunyaQA_alpaca.jsonl # SFT instruction-tuning format (Alpaca)
├── docs/                       # Official clinical guidelines source documents (Markdown)
│   ├── Manejo_Chikungunya_2ed.md
│   ├── IXCHIQ (vacina chikungunya) novo registro.md
│   ├── Nota Tecnica n 28SESSUBPAS-SRAS-DATE-CMI2023.md
│   ├── Nota Tecnica n 642025-CGFAMDPNISVSAMS.md
│   ├── Nota Tecnica n 92026-CGFAMDPNISVSAMS.md
│   └── SBR- Recomentacoes fisioterapeuticas.md
├── pipeline/                   # Core Python modules for SDG, RAG & G-Eval
│   ├── data_ingestion.py       # Structure-aware splitting (MarkdownHeaderTextSplitter)
│   ├── qa_generator.py         # Multi-persona generator agent (GPT-4o mini)
│   ├── evaluator.py            # G-Eval LLM-as-a-Judge (Claude Sonnet 4.5)
│   ├── llm.py                  # API wrapper client implementations
│   └── prompts.py              # Single source of truth prompt templates
├── scripts/                    # Reproduction execution script
│   └── generate_gold_dataset.py # Main reproduction pipeline execution script
├── .env.example                # Template for environment variables
├── requirements.txt            # Python dependencies
└── README.md                   # Repository documentation
```

---

## Environment Setup & Installation

### 1. Clone Repository & Install Dependencies

```bash
git clone https://github.com/denivalpujuca/ChikungunyaQA.git
cd ChikungunyaQA
pip install -r requirements.txt
```

### 2. Configure API Credentials

Copy `.env.example` to `.env` and insert your OpenAI and Anthropic API keys:

```bash
cp .env.example .env
```

Edit `.env`:
```env
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

---

## Reproduction Guide

To reproduce the dataset creation, audit, and evaluation pipeline end-to-end:

### Execute Full Dataset Generation & Dual Audit
Executes data ingestion, multi-persona QA generation via GPT-4o mini (temperature = 0.0) in iterative saturation mode (up to 6 passes per fragment), followed by automated G-Eval auditing via Claude Sonnet 4.5 (cutoff score $\ge$ 90/100):
```bash
python scripts/generate_gold_dataset.py
```

---

## Dataset Specifications

### 1. Full Research Format (`data/ChikungunyaQA.jsonl`)
Contains complete multidimensional provenance metadata:
```json
{
  "id": "chikungunya_qa_0001",
  "persona": "Paciente",
  "question": "Quem deve tomar a vacina IXCHIQ?",
  "answer": "A vacina IXCHIQ é indicada para pessoas com 18 anos ou mais que estejam em risco aumentado de exposição ao vírus chikungunya.",
  "context": "A vacina é indicada IXCHIQ para a prevenção da doença causada pelo vírus da chikungunya...",
  "disease_phase": "N/A (Geral/Prevenção)",
  "tags": ["#Vacinação", "#Chikungunya"],
  "source_file": "IXCHIQ (vacina chikungunya) novo registro.md",
  "chunk_index": 0,
  "pass_num": 1,
  "judge_score": 98.0,
  "judge_details": {
    "G1_faithfulness": 100,
    "G2_completeness": 100,
    "G3_persona_voice": 100,
    "G4_relevance": 95,
    "reason": "Resposta fiel ao trecho da diretriz..."
  }
}
```

### 2. Alpaca Instruction-Tuning Format (`data/ChikungunyaQA_alpaca.jsonl`)
Preprocessed instruction format ready for direct SFT fine-tuning (LoRA, QLoRA, Unsloth, LLaMA-Factory):
```json
{
  "instruction": "Quem deve tomar a vacina IXCHIQ?",
  "input": "A vacina é indicada IXCHIQ para a prevenção da doença...",
  "output": "A vacina IXCHIQ é indicada para pessoas com 18 anos ou mais...",
  "persona": "Paciente"
}
```

---

## FAIR Principles Compliance

ChikungunyaQA strictly adheres to FAIR data principles:
- **Findable:** Permanent DOI [`10.5281/zenodo.20444766`](https://doi.org/10.5281/zenodo.20444766) and structured metadata schemas.
- **Accessible:** Publicly hosted under open CC BY 4.0 license.
- **Interoperable:** Standard JSONL and CSV formats compatible with Hugging Face Datasets, Unsloth, and Pandas.
- **Reusable:** Full clinical context provenance, header hierarchy, and expert validation metadata.

---

## Citation

If you use ChikungunyaQA or code from this repository, please cite our paper:

```bibtex
@article{chikungunyaqa2026,
  title={ChikungunyaQA: A Clinical Dataset of Questions and Answers on Chikungunya},
  author={dos Santos, Denival Araujo and Moreira, Rayele and da Costa, Ana Cristina Vieira and de Barros, Gabriel Martins and Santos, Bruno Sampaio and Teles, Ariel Soares},
  journal={Data},
  year={2026},
  doi={10.5281/zenodo.20444766}
}
```

---

## License

- **Dataset:** [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)
- **Source Code:** [MIT License](LICENSE)
