# Vector DB Benchmark

A benchmarking tool for comparing vector database performance across different databases, embedding models, and document types. Designed to run locally by default with optional AWS/S3 integration.

## Overview

This tool helps you:
- **Benchmark multiple vector databases** (ChromaDB, Qdrant, Milvus Lite, LanceDB, S3 Vectors)
- **Test different embedding models** (local sentence-transformers or AWS Bedrock)
- **Measure end-to-end performance** including ingestion, search, and LLM completion
- **Compare results** across configurations with detailed metrics
- **Run locally** without any cloud dependencies, or scale to S3 for larger workloads

### Key Features

**Local-first design** - Works out of the box without AWS credentials

**Flexible storage** - Local filesystem or S3 for documents and results

**Extensible** - Easy to add new databases, models, or metrics

**Comprehensive metrics** - Tracks latency, throughput, accuracy, and resource usage

## Repository Structure

```
vector-db-benchmark/
├── .env.example                       # Environment configuration template
├── .gitignore                         # Git ignore patterns
├── README.md                          # This file
├── pyproject.toml                     # Python dependencies and project metadata
│
├── data/                              # Local data directory (gitignored)
│   ├── documents/                     # Input documents (PDFs)
│   ├── results/                       # Benchmark results (JSON)
│   └── consolidated_results/          # Aggregated analysis (CSV/Parquet)
│
└── vector_db_benchmark/
    ├── config.py                      # Centralized configuration (S3, AWS, paths)
    │
    ├── benchmark/                     # Core benchmarking logic
    │   ├── main.py                    # CLI entry point
    │   ├── runner.py                  # Job execution orchestration
    │   ├── tasks.py                   # Individual benchmark tasks
    │   └── schemas.py                 # BenchmarkJob dataclass
    │
    ├── databases/                     # Vector database implementations
    │   ├── base.py                    # Abstract VectorDatabase interface
    │   ├── config.py                  # Database-specific config
    │   ├── factory.py                 # Database provider factory
    │   ├── chroma_db.py               # ChromaDB implementation
    │   ├── qdrant_db.py               # Qdrant implementation
    │   ├── milvus_lite_db.py          # Milvus Lite implementation
    │   ├── lance_db.py                # LanceDB implementation
    │   └── s3_vectors_db.py           # AWS S3 Vectors implementation
    │
    ├── embedding/                     # Embedding model providers
    │   ├── base.py                    # Abstract EmbeddingProvider interface
    │   ├── config.py                  # Model catalog (local & Bedrock models)
    │   ├── factory.py                 # Embedder factory
    │   ├── local_embedder.py          # sentence-transformers models
    │   └── bedrock_embedder.py        # AWS Bedrock embeddings
    │
    ├── completion/                    # LLM completion providers
    │   ├── base.py                    # Abstract CompletionProvider interface
    │   ├── config.py                  # Model catalog (Bedrock LLMs)
    │   ├── factory.py                 # Completion provider factory
    │   └── bedrock.py                 # AWS Bedrock completions
    │
    ├── services/                      # Infrastructure services
    │   ├── aws.py                     # AWS client (S3, Bedrock)
    │   └── storage.py                 # FileManager (local/S3 abstraction)
    │
    └── utils/                         # Utility modules
        ├── consolidate_results.py     # Result aggregation (local/S3)
        ├── document_manager.py        # Document discovery
        ├── text_extractor.py          # PDF text extraction
        ├── query_generator.py         # Test query generation
        ├── metrics.py                 # Performance metrics
        ├── logging_config.py          # Logging setup
        └── visualize_benchmark.py     # Result visualization
```

## Configuration Architecture

This project uses a **three-tier configuration system**:

### 1. Runtime Configuration (`vector_db_benchmark/config.py`)
**Purpose:** Environment-specific settings (S3 buckets, AWS regions, file paths)

- Single source of truth for all runtime configuration
- Loaded from environment variables (`.env` file)
- Local-first defaults (no AWS required)
- Used by: all modules throughout the package

**Key settings:**
- `default_docs_dir` → `./data/documents`
- `default_results_dir` → `./data/results`
- `s3_bucket` → For S3 storage mode
- `s3_vector_bucket` → For S3 Vectors database
- `aws_region` → AWS region for S3/Bedrock

### 2. Model Catalog Configuration
**Purpose:** Define available models and their specifications

**`vector_db_benchmark/embedding/config.py`** - Embedding models
- Local models: sentence-transformers from HuggingFace
- Bedrock models: AWS-managed embeddings
- Includes: model name, dimensions, normalization settings
- Add your own models by following the documented template

**`vector_db_benchmark/completion/config.py`** - LLM completion models
- Bedrock models: Claude, Llama, etc.
- Includes: model ID, provider configuration
- Add your own models by following the documented template

### 3. Database Configuration (`vector_db_benchmark/databases/config.py`)
**Purpose:** Database-specific connection settings

- Host/port for local databases (Qdrant, etc.)
- S3 Vectors bucket/region (pulls from centralized config)
- Extensible for future database-specific needs

### Environment Variables (`.env` file)

All optional - only needed for S3/AWS features:

```bash
# S3 storage mode
S3_BUCKET=my-benchmark-bucket

# S3 Vectors database
S3_VECTOR_BUCKET=my-vector-bucket
S3_VECTOR_REGION=us-east-1

# AWS configuration
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=your-key
AWS_SECRET_ACCESS_KEY=your-secret
```

## Quick Start (Local Mode - No AWS Required)

The benchmark runs entirely locally by default. No AWS account or credentials needed!

### 1. Install Dependencies

**Option A: Using uv (recommended - faster)**
```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv pip install -e .
```

**Option B: Using pip**
```bash
pip install -e .
```

> **Note:** For local embedding models (sentence-transformers), also install the optional dependency:
> ```bash
> uv pip install -e ".[local_models]"  # with uv
> pip install -e ".[local_models]"     # with pip
> ```

### 2. Prepare Your Documents

Create a local documents directory and add your PDF files:

```bash
mkdir -p data/documents
# Copy your PDF files to data/documents/
```

### 3. Run Your First Benchmark

```bash
vector-db-benchmark --docs sample --dbs chroma --models local
```

Or using Python module syntax:
```bash
python -m vector_db_benchmark.benchmark.main --docs sample --dbs chroma --models local
```

This will:
- Use sample documents from your local directory
- Benchmark ChromaDB vector database
- Use local embedding models (no API calls)
- Store results in `data/results/`

### 4. View Results

Results are saved as JSON files in `data/results/run_<timestamp>/`

To consolidate results into CSV:

```bash
python -m vector_db_benchmark.utils.consolidate_results --output-dir data/analysis
```

## Configuration Options

### Command Line Arguments

```bash
vector-db-benchmark \
  --docs sample              # Document selection: sample, all, small, medium, large
  --dbs chroma qdrant        # Vector databases to test
  --models local             # Embedding models: local, bedrock, all
  --llms none                # LLM models: claude-3-sonnet, all, none (default: none)
  --runs 3                   # Number of runs per configuration
  --storage-mode local       # Storage: local or s3 (default: local)
  --docs-dir ./data/documents    # Documents directory
  --results-dir ./data/results   # Results directory
```

### Available Databases

- `chroma` - ChromaDB (local)
- `qdrant` - Qdrant (local)
- `milvus_lite` - Milvus Lite (local)
- `lance` - LanceDB (local)
- `s3_vectors` - AWS S3 Vectors (requires AWS)

### Available Embedding Models

**Local models** (no AWS required):
- `minilm` - Lightweight (384 dimensions) - `sentence-transformers/all-MiniLM-L6-v2`
- `mpnet` - Higher quality (768 dimensions) - `sentence-transformers/all-mpnet-base-v2`
- `local` - Use all local models

**Bedrock models** (requires AWS):
- `titan-v2` - Amazon Titan Embeddings v2 (512 dimensions)
- `bedrock` - Use all Bedrock models

**Add your own:** Edit `vector_db_benchmark/embedding/config.py` following the documented template

### Available Completion Models

**Bedrock models** (requires AWS):
- `claude-3-sonnet` - Claude 3.5 Sonnet (cross-region inference profile)
- `llama3-8b` - Meta Llama 3 8B Instruct
- `all` - Use all completion models
- `none` - Skip completion step (default)

**Add your own:** Edit `vector_db_benchmark/completion/config.py` following the documented template

## Advanced: S3/AWS Mode (Optional)

If you want to use AWS services for remote storage or AWS-specific databases:

### 1. Set Up AWS Credentials

Configure AWS credentials using one of these methods:

```bash
# Option 1: AWS CLI
aws configure

# Option 2: Environment variables
export AWS_ACCESS_KEY_ID=your-key
export AWS_SECRET_ACCESS_KEY=your-secret
export AWS_REGION=us-east-1

# Option 3: Use .env file (recommended)
cp .env.example .env
# Edit .env and add your AWS configuration
```

### 2. Configure S3 in .env

Create a `.env` file from the example:

```bash
cp .env.example .env
```

Edit `.env` and configure:

```bash
# General S3 bucket for benchmark storage
S3_BUCKET=my-benchmark-bucket

# S3 Vectors database bucket (if using S3 Vectors)
S3_VECTOR_BUCKET=my-vector-bucket
S3_VECTOR_REGION=us-east-1

# AWS region
AWS_REGION=us-east-1
```

### 3. Run with S3 Storage

```bash
# Store databases in S3
vector-db-benchmark \
  --storage-mode s3 \
  --docs-dir s3://my-bucket/documents \
  --results-dir s3://my-bucket/results

# Use S3 Vectors database
vector-db-benchmark \
  --dbs s3_vectors \
  --models bedrock
```

## Environment Variables Reference

All configuration via `.env` file:

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `S3_BUCKET` | No | - | S3 bucket for benchmark storage (only for S3 mode) |
| `S3_VECTOR_BUCKET` | No | - | S3 bucket for S3 Vectors database |
| `S3_VECTOR_REGION` | No | us-east-1 | AWS region for S3 Vectors |
| `AWS_REGION` | No | us-east-1 | AWS region for S3 and Bedrock |
| `AWS_ACCESS_KEY_ID` | No | - | AWS access key (or use AWS CLI/IAM) |
| `AWS_SECRET_ACCESS_KEY` | No | - | AWS secret key (or use AWS CLI/IAM) |

## Troubleshooting

### "S3_BUCKET environment variable must be set"

This error occurs when using `--storage-mode s3` without configuring S3_BUCKET.

**Solution**: Either:
1. Use local mode: `--storage-mode local` (default)
2. Set S3_BUCKET in your `.env` file

### "Failed to initialize DocumentManager"

Your documents directory doesn't exist or is empty.

**Solution**:
```bash
mkdir -p data/documents
# Add PDF files to data/documents/
```

### AWS Credentials Errors

If you get AWS credential errors when using S3 or Bedrock:

**Solution**:
1. Run `aws configure` to set up AWS CLI credentials
2. Or set AWS environment variables in `.env`

## Examples

### Basic Local Benchmark
```bash
# Benchmark all local databases with local models
vector-db-benchmark \
  --docs all \
  --dbs all \
  --models local \
  --runs 3
```

### AWS Bedrock Benchmark
```bash
# Use AWS Bedrock for embeddings and completions
vector-db-benchmark \
  --docs sample \
  --dbs chroma qdrant \
  --models bedrock \
  --llms claude-3-sonnet
```

### S3 Storage Mode
```bash
# Store everything in S3
vector-db-benchmark \
  --storage-mode s3 \
  --docs-dir s3://my-bucket/documents \
  --results-dir s3://my-bucket/results \
  --dbs all
```

### List Available Documents
```bash
# See what documents are available
vector-db-benchmark --list-docs
```

## How It Works

### Benchmark Pipeline

Each benchmark job follows a three-phase pipeline:

```
1. INGESTION PHASE
   ├─ Extract text from PDF documents
   ├─ Chunk text (configurable size/overlap)
   ├─ Generate embeddings (local or Bedrock)
   ├─ Store vectors in database
   └─ Measure: latency, throughput, storage size

2. SEARCH PHASE
   ├─ Generate test queries from documents
   ├─ Embed queries
   ├─ Perform vector similarity search
   ├─ Retrieve top-k results
   └─ Measure: query latency, recall, precision

3. COMPLETION PHASE (optional)
   ├─ Use search results as context
   ├─ Generate completion with LLM
   ├─ Measure: completion latency, token usage
   └─ Save final results
```

### Storage Modes

**Local Mode** (default):
- Databases stored in `/tmp/vector_db_benchmark/`
- Results in `data/results/`
- No AWS credentials needed
- Great for development and testing

**S3 Mode** (optional):
- Databases persisted to `s3://{S3_BUCKET}/databases/`
- Results to `s3://{S3_BUCKET}/results/`
- Enables sharing results across team
- Required for S3 Vectors database

### Result Files

Each run produces JSON files with detailed metrics:

```
data/results/run_1234567890/
├── ingestion_chroma_minilm_document1.json
├── search_chroma_minilm_document1.json
├── ingestion_qdrant_mpnet_document2.json
└── search_qdrant_mpnet_document2.json
```

Consolidate results:
```bash
python -m vector_db_benchmark.utils.consolidate_results \
  --results-path data/results \
  --output-dir data/analysis \
  --format csv
```

## Extending the Benchmark

### Adding a New Vector Database

1. Create `vector_db_benchmark/databases/your_db.py`:
```python
from vector_db_benchmark.databases.base import VectorDatabase

class YourDB(VectorDatabase):
    def add_vectors(self, vectors, metadata): ...
    def search(self, query_vector, top_k): ...
    # Implement abstract methods
```

2. Register in `vector_db_benchmark/databases/factory.py`:
```python
PROVIDER_CLASSES = {
    "your_db": YourDB,
    # ... existing databases
}
```

3. Use it: `--dbs your_db`

### Adding a New Embedding Model

Edit `vector_db_benchmark/embedding/config.py`:
```python
MODEL_CONFIGS = {
    "your-model": ModelConfig(
        name="sentence-transformers/your-model-name",
        provider="local",
        dimensions=1024,
    ),
}
```

Use it: `--models your-model`

### Adding a New Completion Model

Edit `vector_db_benchmark/completion/config.py`:
```python
COMPLETION_MODEL_CONFIGS = {
    "your-llm": CompletionModelConfig(
        name="your-llm",
        model_id="your.bedrock.model.id",
        provider="bedrock",
    ),
}
```

Use it: `--llms your-llm`


## Contributing

To contribute:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

For issues or questions:
- Open an issue on GitHub
- Check existing issues for solutions
- Review the troubleshooting section above

## Next Steps

- ✅ Run your first local benchmark
- 📊 Analyze results with consolidation script
- 🔬 Compare different vector databases
- 📈 Visualize performance metrics
- 🚀 Scale to S3 for production workloads
