# Segment-Graph Retrieval Memory (SGRM)

**Extend the context window of any Large Language Model (LLM) by building a retrieval-augmented, graph-structured memory over long documents.**

---

## Overview

Modern Large Language Models are limited by their fixed token context window, making them incapable of reasoning over long texts or recalling distant information. **Segment-Graph Retrieval Memory (SGRM)** addresses this limitation by:

- **Chunking** long documents into manageable segments.
- **Embedding** each chunk into a semantic vector space.
- **Building a dynamic, directed graph** where nodes are segments and edges encode both chronology and semantic similarity.
- **Retrieving relevant history** via graph search, given any user query.
- **(Optionally) Summarizing** retrieved context to fit LLM input limits.
- **Augmenting LLM prompts** with the most relevant past context, enabling long-range reasoning and QA.

---

## Features

- **Plug-and-play architecture**: Modular Python classes for chunking, embedding, graph management, retrieval, summarization, and LLM prompting.
- **Fast and efficient**: Uses sentence-transformer embeddings and similarity search for rapid context lookup.
- **Flexible backends**: Supports Hugging Face Transformers for both LLMs and summarizers.
- **Scalable**: Handles input far beyond the native context window of GPT-2/3 and similar models.

---

## Installation

1. **Clone the repo:**
    ```bash
    git clone https://github.com/alakhsharma22/segment-graph-memory.git
    cd segment-graph-memory
    ```

2. **Install dependencies:**
    ```bash
    pip install torch transformers sentence-transformers scikit-learn networkx
    ```

---

## Usage

1. **Prepare your input:**
    - Place a long text document in `input.txt` or any path of your choosing.

2. **Run the pipeline:**
    ```bash
    python main.py --text_file input.txt --query "What breakthroughs in battery chemistry enabled modern EVs?" --summarize
    ```

    - `--chunk_size 512` sets chunk size (in tokens).
    - `--k 3` controls number of similarity edges.
    - `--top_n 3` retrieves top-N context paths.
    - `--model_name gpt2` sets the LLM for generation.
    - `--summ_model facebook/bart-large-cnn` sets the summarizer model.
    - `--summarize` enables context summarization (recommended).

3. **Example output:**
    ```
    --- Generated Output ---
    Modern EVs are enabled by advances in lithium-ion battery chemistry...
    ```

---

## Configuration

You can adjust:
- **Summarization model:** (`--summ_model`)
- **LLM model:** (`--model_name`)
- **Chunk size and k:** (`--chunk_size`, `--k`)
- **Hardware backend:** Use CPU, CUDA, or Apple MPS as appropriate.

---

## Mathematical Foundation

SGRM is built on rigorous mathematical principles:
- **Cosine similarity:** Finds semantic neighbors in embedding space.
- **Graph theory:** Enables multi-hop context retrieval, balancing local and global history.
- **Complexity:** Scales efficiently with approximate nearest neighbor search (FAISS/Annoy).

For details, see the [project paper (LaTeX source)](./paper.tex).

---

## Example: input.txt

A ready-to-use example input file on electric vehicles is provided. Modify or replace as needed.
