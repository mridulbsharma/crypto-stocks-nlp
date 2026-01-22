# AGENTS.md - CryptoSentinel Project Guidance

## Project Overview
CryptoSentinel is an agentic AI system for grounded crypto/stock market intelligence.
It transforms Reddit NLP analysis into a production-grade system with MCP tools,
self-Socratic reasoning, and evidence-based verification.

## Directory Structure
```
crypto-stocks-nlp/
├── src/                      # Core library
│   ├── context_engine/       # Semantic search & evidence management
│   ├── mcp_server/           # MCP protocol implementation
│   ├── reasoning/            # Socratic refinement & personas
│   ├── models/               # ML classifiers (NB+LR ensemble)
│   └── utils/                # Text preprocessing
├── app/                      # Application layer
│   ├── streamlit_app.py      # Interactive demo UI
│   └── api.py                # FastAPI REST endpoints
├── evals/                    # Evaluation suite
│   ├── groundedness.py       # Citation accuracy tests
│   └── refusal.py            # Proper refusal tests
├── data/                     # Reddit post data (CSV)
├── code/                     # Original Jupyter notebooks
└── presentation/             # Project presentation
```

## Commands

### Run Streamlit Demo
```bash
cd crypto-stocks-nlp
streamlit run app/streamlit_app.py
```

### Run FastAPI Server
```bash
cd crypto-stocks-nlp
uvicorn app.api:app --reload --port 8000
```

### Run MCP Server (stdio)
```bash
cd crypto-stocks-nlp
python -m src.mcp_server.server
```

### Run Evaluations
```bash
cd crypto-stocks-nlp
python -m evals.groundedness
python -m evals.refusal
```

### Run Tests
```bash
cd crypto-stocks-nlp
pytest evals/ -v
```

## Code Style
- Type hints on all function signatures
- Docstrings for all public classes/functions
- snake_case for variables/functions, PascalCase for classes
- Use dataclasses for data structures
- Maximum line length: 100 characters

## Key Components

### Context Engine
- `PostRetriever`: Semantic search over Reddit posts
- `EmbeddingEngine`: TF-IDF or transformer embeddings
- `EvidenceBudget`: Manages citation requirements

### MCP Tools
- `reddit_search`: Search posts with filters
- `sentiment_aggregate`: Aggregate topic sentiment
- `topic_clusters`: Extract discussion themes
- `classify_post`: Crypto vs stocks classification
- `generate_brief`: Grounded narrative with citations

### Reasoning
- `SocraticRefiner`: Plan → Execute → Verify → Finalize loop
- `DomainAnalyst`: Creative analysis persona
- `EvidenceVerifier`: Strict verification with veto power

## Data Files
- `data/combined_df.csv`: Main dataset (~5500 posts)
- `data/wsb_merged_posts.csv`: WallStreetBets posts
- `data/cms_merged_posts.csv`: CryptoMoonShots posts

## Environment Setup
```bash
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

## API Endpoints
- `GET /health` - Health check
- `POST /search` - Search Reddit posts
- `POST /sentiment` - Sentiment aggregation
- `POST /classify` - Post classification
- `POST /brief` - Generate grounded brief
- `POST /reason` - Full Socratic reasoning
- `GET /clusters` - Topic clusters
- `GET /tools` - MCP tool definitions
