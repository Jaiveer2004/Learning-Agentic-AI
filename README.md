# Learning Agentic AI

A practical learning repository covering the foundations used to build
generative-AI and agentic applications with Python. The material progresses
from Python project setup and APIs to data engineering, NLP preprocessing,
text vectorization, and the deep-learning architectures behind modern
language models.

## Repository roadmap

### Stage 0 - Python APIs and Gemini integration

- **`api-demo`**: a small Google Gemini interaction using the `google-genai`
  SDK. It loads an API key from an environment variable and prints the model's
  response.
- **`awesome-project`**: a minimal FastAPI application demonstrating:
  - a health-style root endpoint (`GET /`)
  - typed path and query parameters (`GET /items/{item_id}`)
  - request-body validation with Pydantic (`PUT /items/{item_id}`)

### Stage 1 - Data and backend foundations

- **`Numpy`**: a small `uv`-managed Python package scaffold for NumPy
  exploration.
- **`Final Project`**: an end-to-end banking data pipeline:
  - `Project_Pandas.ipynb` generates customer and transaction data, cleans and
    validates types, joins relational data, aggregates features, persists the
    result in SQLite, and demonstrates a FastAPI service.
  - `Project_Polars.ipynb` explores schema validation, lazy loading,
    deduplication, expression-based transformations, and temporal
    standardization with Polars.
  - `Project Code/` contains the standalone SQLite/FastAPI implementation.
  - CSV files contain raw customer and transaction data, including orphan
    transactions for data-quality exercises.

### Stage 3 - NLP and deep-learning foundations

- **Text preprocessing 1**:
  - tokenization
  - stemming
  - lemmatization
- **Text preprocessing 2**:
  - preprocessing exercises
  - part-of-speech tagging
  - named-entity recognition
- **Text-to-vectors**:
  - one-hot encoding
  - Bag of Words
  - n-grams
  - TF-IDF
  - word embeddings
  - Word2Vec
- **Encoder-Decoder**:
  - neural-network and ANN foundations
  - Simple RNN, LSTM, GRU, and Bidirectional RNN
  - synchronized and encoder-decoder Many-to-Many architectures
  - Seq2Seq encoder, context vector, decoder, and training workflow
  - teacher forcing, beam search, and context-vector bottlenecks
- **Attention Mechanism**:
  - attention scores, weights, and context vectors
  - additive and multiplicative attention
  - Query, Key, and Value representations
  - scaled dot-product and multi-head attention
  - self-attention, masked attention, and cross-attention
  - positional encoding, padding masks, causal masks, KV cache, MQA,
    and GQA
- **Transformers**:
  - Transformer encoder and decoder architecture
  - token embeddings and positional information
  - residual connections, layer normalization, and feed-forward networks
  - encoder-only, decoder-only, and encoder-decoder models
  - autoregressive generation and Transformer implementation with PyTorch

## Directory structure

```text
.
├── Stage 0/
│   ├── api-demo/
│   └── awesome-project/
├── Stage 1/
│   ├── Final Project/
│   │   ├── Project Code/
│   │   ├── Project_Pandas.ipynb
│   │   └── Project_Polars.ipynb
│   └── Numpy/
└── Stage 3/
    ├── 01_Text-Pre-Processing-1/
    ├── 02_Text-Pre-Processing-2/
    ├── 03_Text-to-Vectors/
    ├── 04_Encoder-Decoder/
    ├── 05_Attention-Mechanism/
    └── 06_Transformers/
```

### Stage 3 reading order

The Markdown guides are intended to be studied in this order:

1. [Text-to-Vectors](<Stage 3/03_Text-to-Vectors/>)
2. [Encoder-Decoder and Seq2Seq](<Stage 3/04_Encoder-Decoder/>)
3. [Attention Mechanism](<Stage 3/05_Attention-Mechanism/>)
4. [Transformers](<Stage 3/06_Transformers/>)

The Encoder-Decoder section builds the recurrent and Seq2Seq foundation.
The Attention section then explains how models learn which representations
to prioritize. The Transformer section combines those ideas into the
attention-first architecture used by modern language models.

The repository also contains local virtual environments used during
development. They are not part of the learning material and should not be
committed.

## Prerequisites

- Python 3.13 or newer
- [uv](https://docs.astral.sh/uv/) for the Stage 0 projects
- Jupyter Notebook or JupyterLab for notebook-based lessons
- An optional Google Gemini API key for `Stage 0/api-demo`

## Running the API examples

### FastAPI example

```powershell
cd "Stage 0/awesome-project"
uv sync
uv run fastapi dev main.py
```

Open `http://127.0.0.1:8000/docs` to explore the generated Swagger UI.

### Gemini API example

Set the API key in `Stage 0/api-demo/.env`:

```text
API_KEY=your_api_key_here
```

Then run:

```powershell
cd "Stage 0/api-demo"
uv sync
uv run python main.py
```

Never commit a real API key. Keep `.env` files local and add them to
`.gitignore` before pushing the repository.

## Running the data project

Install the notebook dependencies from the final project directory:

```powershell
cd "Stage 1/Final Project"
python -m pip install -r requirements.txt
```

Launch Jupyter from the repository root or the project directory:

```powershell
jupyter notebook
```

Run `Project_Pandas.ipynb` first for the complete Pandas-to-SQLite workflow.
`Project_Polars.ipynb` is a separate performance-oriented exploration.

The standalone API can be started from its code directory:

```powershell
cd "Stage 1/Final Project/Project Code"
python -m pip install fastapi "uvicorn[standard]"
uvicorn main:app --reload
```

Available endpoints:

- `GET /customers/{customer_id}` - retrieve a customer profile
- `GET /transactions` - retrieve transactions with optional `status`,
  `limit`, and `offset` parameters

The service uses the SQLite database in `Project Code/bank_data.db`.

## Learning goals

By working through the stages, you will practice:

1. Managing modern Python projects and dependencies.
2. Calling a generative-AI model from Python.
3. Building typed HTTP APIs with FastAPI and Pydantic.
4. Generating, cleaning, validating, joining, and storing data.
5. Comparing Pandas and Polars data-processing workflows.
6. Preparing text for machine-learning pipelines.
7. Converting text into numerical representations for downstream models.
8. Understanding how RNN-based encoder-decoder systems process sequences.
9. Explaining attention mathematically using Q, K, V, scores, weights, and
   context vectors.
10. Understanding Transformer encoders, decoders, masking, positional
    encoding, and autoregressive generation.
11. Implementing scaled dot-product attention and Transformer components
    with PyTorch.

## Notes

- The notebooks are the primary learning material and may contain exploratory
  code intended to be run cell by cell.
- Dataset files and SQLite databases are included for reproducibility and API
  practice.
- Dependency lock files are retained for the `uv`-managed Stage 0 projects.
- The Stage 3 Markdown guides include local diagrams in each topic's `images/`
  directory so they render reliably in repository previews.
- Equations use Markdown-compatible LaTeX display blocks, and implementation
  examples use fenced Python code blocks for readable previews.
