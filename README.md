# AI-Powered Data Matching

## 🚀 Overview

This project implements a scalable, intelligent data matching pipeline using LLMs and vector-based retrieval. The goal is to match records from `input.csv` to the most appropriate entries in `org_data.csv` and `factor_data.csv`, outputting a result that closely aligns with `result.csv`.

We leverage OpenAI's GPT-4o and sentence transformers to integrate retrieval-augmented generation (RAG) for efficient and accurate decision-making.

---

## 📁 Project Structure

```
├── data
    ├── input.csv                  # Source data to match
    ├── org_data.csv               # Organization reference data
    ├── factor_data.csv            # Additional matching factors
    ├── result.csv                 # Ground truth for validation
├── results.csv                # Generated output (your submission)
├── pipeline.py                # Main execution script
├── matcher.py                 # Matching logic using LLM + retrieval
├── retriever.py               # Embedding and top-k retrieval
├── llm_utils.py               # OpenAI API integration
└── README.md                  # Project documentation
```

---

## 🧠 Core Features

- 🔗 **LLM-powered Matching**: Uses GPT-4 to intelligently match supplier records.
- 🔍 **RAG Design**: Retrieves top-k relevant records via semantic search (using `sentence-transformers`).
- ⚙️ **Modular Pipeline**: Easy to extend, test, or parallelize.
- 📈 **Scalable Architecture**: Works efficiently even with large datasets.

---

## 📦 Requirements

- Python 3.8+
- [OpenAI API Key](https://platform.openai.com/account/api-keys)

### Install Dependencies

```bash
pip install -r requirements.txt
```

<details>
<summary>Example <code>requirements.txt</code></summary>

```
pandas
openai
sentence-transformers
torch
```

</details>

---

## 🔑 Setup

1. Place your `input.csv`, `org_data.csv`, and `factor_data.csv` in the project root.
2. Set your OpenAI API key in `matcher.py` or using an environment variable:

```bash
export OPENAI_API_KEY="your-api-key-here"
```

3. Run the pipeline:

```bash
python3 main.py
```

4. The output will be saved to `results.csv`.

---

## 📊 Output Format

The `results.csv` will contain the following fields:

- `Subcategory`
- `Subcategory 2`
- `Supplier`
- `Organization`
- `Sectors`
- `Factor Id`
- `Factor Name`

---


## 📌 Design Notes

- Top-k matching reduces prompt size and cost.
- Using `sentence-transformers` enables semantic understanding of context.
- LLM prompt is templated to include only the most relevant subset of orgs and factors.
- Results can be post-validated using fuzzy matching or scoring heuristics.

---
