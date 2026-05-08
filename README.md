# ChikungunyaQA: Clinical Question-Answering Dataset

This repository contains the **ChikungunyaQA** dataset and the reproduction scripts associated with the paper: *"ChikungunyaQA: A High-Fidelity, Expert-Validated Question-Answering Dataset Derived from Clinical Guidelines"*.

## 📁 Repository Structure

*   `data/`: Contains the official gold-standard dataset (`chikungunya_qa_gold.jsonl`).
*   `docs/`: Source clinical guidelines (in Markdown) used to generate the dataset.
*   `scripts/`:
    *   `data_ingestion.py`: Logic for parsing and chunking medical documents.
    *   `validacao_humana.py`: Script to calculate inter-rater agreement (Cohen's Kappa).
*   `pipeline/`: Core RAG and G-Eval architecture used for data generation and auditing.

## 📊 Dataset Overview

ChikungunyaQA consists of **1,078** expert-validated question-answering pairs. Every pair is grounded in official clinical manuals from the Brazilian Ministry of Health.

| Metric | Value |
| :--- | :--- |
| Total QA Pairs | 1078 |
| Language | Portuguese (PT-BR) |
| Validation | G-Eval (LLM-as-a-Judge) + Human Expert Review |
| Domains | Diagnosis, Treatment, Risk Groups, Chronic Phase, Vaccination |

## 🚀 Getting Started

### 1. Installation

```bash
pip install -r requirements.txt
```

### 2. Accessing the Data

You can load the dataset using Python:

```python
import pandas as pd
import json

data = []
with open('data/chikungunya_qa_gold.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        data.append(json.loads(line))

df = pd.DataFrame(data)
print(df.head())
```

### 3. Reproducibility Note

Due to the stochastic nature of Large Language Models (LLMs), re-running the generation pipeline may result in slight variations in wording. However, the methodology ensures consistent clinical accuracy and semantic density. The versioned models used in our study are:
*   **Generator:** GPT-4o-mini-2024-07-18
*   **Auditor:** Claude-3-5-Sonnet-20240620 (G-Eval)

## ⚖️ License

This dataset is licensed under the [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/) license.

## ✍️ Citation

If you use this dataset in your research, please cite:

```bibtex
@article{denival2026chikungunyaqa,
  title={ChikungunyaQA: A High-Fidelity, Expert-Validated Question-Answering Dataset Derived from Clinical Guidelines},
  author={Denival, et al.},
  journal={MDPI Data},
  year={2026}
}
```
