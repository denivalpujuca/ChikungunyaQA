# ChikungunyaQA: A Clinical Question-Answering Dataset

[![Zenodo DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21908616.svg)](https://doi.org/10.5281/zenodo.21908616)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)

This repository provides the dataset files and the reproduction pipeline for **ChikungunyaQA**, a clinical multi-persona QA dataset grounded in Brazilian official guidelines and regulatory documents.

> **For detailed information on the dataset construction methodology, prompt engineering, automated G-Eval auditing, expert clinical validation, and statistical analyses, please refer to the full paper:**  
> **ChikungunyaQA: A Clinical Dataset of Questions and Answers on Chikungunya** (*Data, MDPI, 2026* — Article Under Review).

---

## Repository Structure

```text
.
├── data/                        # ChikungunyaQA dataset files
│   ├── ChikungunyaQA.jsonl      # Full research format with complete metadata
│   ├── ChikungunyaQA.csv        # Full research format (CSV)
│   └── ChikungunyaQA_alpaca.jsonl # SFT instruction-tuning format (Alpaca)
├── docs/                        # Official clinical source guidelines (Markdown)
├── pipeline/                    # Reproduction pipeline modules
│   ├── data_ingestion.py        # Structure-aware heading segmentation
│   ├── qa_generator.py          # Inductive generator agent, saturation loop & G-Eval audit
│   ├── llm.py                   # Model API connectors (GPT-4o mini & Claude Sonnet 4.5)
│   └── prompts.py               # Prompt templates
├── scripts/
│   └── generate_gold_dataset.py # End-to-end reproduction script
├── .env.example                 # Environment variables template
├── requirements.txt             # Python dependencies
└── README.md
```

---

## How to Reproduce

### 1. Installation

```bash
git clone https://github.com/denivalpujuca/ChikungunyaQA.git
cd ChikungunyaQA
pip install -r requirements.txt
```

### 2. Configure API Keys

Copy `.env.example` to `.env` and set your API credentials:

```bash
cp .env.example .env
```

```env
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

### 3. Run Pipeline

Run the generation and audit pipeline:

```bash
python scripts/generate_gold_dataset.py
```

Generated outputs will be saved to the `output/` folder.

---

## Dataset Access

The validated dataset is available in the `data/` directory and on **[Zenodo (DOI: 10.5281/zenodo.21908616)](https://doi.org/10.5281/zenodo.21908616)**:

* **`ChikungunyaQA.jsonl` / `ChikungunyaQA.csv`**: Full research version containing clinical metadata (disease phase, tags, chunk indices) and G-Eval scores with judge rationales.
* **`ChikungunyaQA_alpaca.jsonl`**: Standard instruction-tuning format (`instruction`, `input`, `output`, `persona`) ready for SFT / LoRA / QLoRA training.

---

## Citation

If you use ChikungunyaQA in your research, please cite:

```text
Article Under Review
```

---

## License

* **Dataset:** [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)
* **Code:** [MIT License](LICENSE)
