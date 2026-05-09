# hacknight-blr-syy-misinfo-deetector

Production-oriented agentic misinformation verifier for Bengaluru civic claims.

The system accepts a user claim, retrieves external evidence, evaluates every source independently, validates citations, computes confidence, optionally reformulates low-confidence searches, and returns a structured JSON verdict.

## Production Architecture

```text
POST /verify
  -> Retrieval Agent
       -> /provide_sources-compatible retrieval provider
       -> Elasticsearch BM25 + ELSER hybrid search
       -> Elastic Jina rerank
  -> Evidence Analyzer Agent
       -> source-by-source LLM reasoning
       -> SUPPORTS / CONTRADICTS / PARTIALLY_SUPPORTS / NOT_MENTIONED
  -> Confidence Agent
       -> agreement + relevance + evidence coverage scoring
  -> Query Reformulation Agent
       -> bounded retry when confidence < 0.5
       -> maximum 2 re-query attempts
  -> Final Verdict Agent
       -> SUPPORTED / UNSUPPORTED / PARTIALLY_SUPPORTED / UNVERIFIED
  -> Validation Agent
       -> strict schema validation
       -> citation id validation
       -> downgrade to UNVERIFIED for empty/invalid citations
```

## Folder Structure

```text
src/agents/              # Modular agent classes for retrieval, analysis, confidence, reformulation, validation, final verdict
src/prompts/             # Secure sandwich prompt templates with system/data/output separation
src/validators/          # Deterministic post-LLM validation, especially citation enforcement
src/schemas/             # Pydantic API, source, assessment, confidence, and verdict models
src/services/            # LLM provider facade; currently Elastic Inference chat/completion compatible
src/orchestration/       # VerificationGraph wiring agents, retry loop, and state transitions
src/retrieval/           # Retrieval provider abstraction over Elasticsearch search/rerank
src/utils/               # Cross-cutting utilities such as structured logging setup
src/configs/             # Reserved for deploy/runtime config extensions
src/api.py               # FastAPI routes: /provide_sources, /verify, legacy /check, /live-check
config/                  # Elasticsearch mappings, ELSER pipeline, trusted source seeds
tests/                   # Unit tests for validation and confidence behavior
logs/                    # Runtime log target for deployments that persist local logs
```

This layout keeps LLM behavior isolated from deterministic validation and retrieval. New retrievers, LLM providers, or verdict policies can be added without changing the API contract.

## API Architecture

### `POST /provide_sources`

Retrieves and normalizes the top evidence documents.

Request:

```json
{
  "claim": "BMRCL announced Purple Line closure on Tuesday.",
  "retrieval_size": 30,
  "top_k": 5
}
```

Response:

```json
[
  {
    "id": "doc_1",
    "url": "https://example.com",
    "text": "Retrieved article content",
    "date_of_scraping": "2026-05-09",
    "title": "Service update",
    "source_name": "BMRCL",
    "source_type": "official",
    "retrieval_score": 8.1,
    "rerank_score": 0.91
  }
]
```

### `POST /verify`

Runs the full agentic verification graph.

Request:

```json
{
  "claim": "BMRCL announced Purple Line closure on Tuesday.",
  "retrieval_size": 30,
  "top_k": 5,
  "max_requery_attempts": 2
}
```

Response:

```json
{
  "claim": "BMRCL announced Purple Line closure on Tuesday.",
  "verdict": "SUPPORTED",
  "confidence": 0.82,
  "summary": "The retrieved sources confirm the service disruption announcement.",
  "citations": [
    {
      "id": "doc_1",
      "url": "https://example.com",
      "date_of_scraping": "2026-05-09"
    }
  ],
  "attempts": 0,
  "analyzed_sources": [
    {
      "source_id": "doc_1",
      "label": "SUPPORTS",
      "rationale": "The source directly describes the announced Purple Line closure.",
      "cited_quote": "Purple Line services will be unavailable on Tuesday...",
      "relevance": 0.92
    }
  ],
  "refined_queries": [],
  "injection_warnings": []
}
```

## Security Design

Retrieved documents and user claims are always treated as untrusted data. The prompt templates in `src/prompts/templates.py` enforce a sandwich structure:

```text
system: security policy and task boundaries
user: USER CLAIM triple-quoted as data
user: RETRIEVED SOURCES triple-quoted as data
user: OUTPUT INSTRUCTIONS and JSON schema
```

The system prompt explicitly states that retrieved documents are data only, instruction-like text in evidence must be ignored, and commands from evidence must never be executed. Evidence text is neutralized with length and null-byte controls before entering prompts.

Deterministic validation happens after the LLM:

- Pydantic validates schema and enum values.
- Citation IDs must exist in the retrieved source set.
- Empty citations or invalid citation IDs automatically downgrade the final verdict to `UNVERIFIED`.
- Prompt-injection pattern matches are surfaced in `injection_warnings` but never followed.

## Agent Responsibilities

- **Retrieval Agent** calls the retrieval provider and normalizes sources into `SourceEnvelope`.
- **Evidence Analyzer Agent** performs independent source reasoning and labels every source.
- **Confidence Agent** scores agreement, relevance, and coverage. Scores below `0.5` trigger retry.
- **Query Reformulation Agent** creates a safer refined retrieval query when evidence is weak.
- **Validation Agent** enforces citation and schema correctness after generation.
- **Final Verdict Agent** synthesizes source assessments into the final structured verdict.

## Retry and Re-query Pipeline

The graph allows at most two re-query attempts:

```text
claim -> retrieve -> analyze -> score
if confidence >= 0.5 -> validate -> return
if confidence < 0.5 -> reformulate query -> retrieve again -> analyze again
after max attempts -> return best validated result
```

If query reformulation repeats a prior query, the graph stops and returns an `UNVERIFIED` result capped below `0.5`.

## Error Handling and Observability

- FastAPI response models enforce API shape at the boundary.
- LLM JSON parse failures are logged and converted into deterministic fallback verdicts.
- Malformed per-source assessments are ignored and replaced with `NOT_MENTIONED` assessments.
- Logs are emitted through Python logging with module names and event messages such as `verification_attempt`, `llm_json_parse_failed`, and validation failures.
- `/health` remains lightweight and does not require Elasticsearch credentials.

## Configuration Management

Runtime configuration is loaded from `.env` via `src/settings.py`.

```text
ELASTIC_CLOUD_ID=
ELASTICSEARCH_URL=
ELASTICSEARCH_API_KEY=
ELASTICSEARCH_INDEX=blr_claim_sources_v1
ELSER_INFERENCE_ID=.elser-2-elasticsearch
ELSER_QUERY_MODE=sparse_vector
JINA_RERANK_INFERENCE_ID=jina-reranker-v3
LLM_INFERENCE_ID=.gp-llm-v2-chat_completion
LLM_TASK_TYPE=chat_completion
LLM_MAX_TOKENS=900
```

Secrets stay in `.env`; `.env.example` documents required keys.

## Prompt Templates

The production prompt templates live in `src/prompts/templates.py`.

They produce role-separated message arrays:

```python
[
    {"role": "system", "content": SECURITY_SYSTEM_PROMPT},
    {"role": "user", "content": "USER CLAIM:\n\"\"\"...\"\"\"\n\nRETRIEVED SOURCES:\n\"\"\"...\"\"\"\n\nOUTPUT INSTRUCTIONS:\n..."}
]
```

This preserves system authority and keeps source text in a literal data block.

## Layer 1: Retrieval and Citation Pipeline

This repo currently contains the first layer for the Bengaluru misinformation checker:

```text
Elastic Open Crawler sources -> Elasticsearch index -> ELSER hybrid search -> Elastic Jina rerank -> Elastic LLM verdict agent -> cited evidence
```

### Files

```text
config/sources.yaml          # trusted seed sources
config/elastic_index.json    # Elasticsearch mapping
config/elser_pipeline.json   # ELSER ingest pipeline
src/create_index.py          # creates ELSER endpoint, pipeline, and index
src/search.py                # BM25 + ELSER hybrid retrieval
src/rerank.py                # Elastic Inference API Jina reranker integration
src/agent.py                 # Elastic Inference API LLM answer generation
src/injection_guard.py       # prompt-injection detection/neutralization helpers
src/claim_check.py           # CLI claim checker
src/api.py                   # FastAPI wrapper
src/crawler_config.py        # writes crawler seed URLs
```

### Setup

Create a virtualenv and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Copy the env template:

```bash
cp .env.example .env
```

Fill these first:

```text
ELASTICSEARCH_URL=
ELASTICSEARCH_API_KEY=
ELASTICSEARCH_INDEX=blr_claim_sources_v1
ELSER_INFERENCE_ID=.elser-2-elasticsearch
ELSER_QUERY_MODE=sparse_vector
JINA_RERANK_INFERENCE_ID=jina-reranker-v3
LLM_INFERENCE_ID=.gp-llm-v2-chat_completion
LLM_TASK_TYPE=chat_completion
LLM_MAX_TOKENS=900
```

At runtime, the app talks only to Elasticsearch using `ELASTICSEARCH_API_KEY`.

Expected Elastic inference endpoints:

```text
_inference/sparse_embedding/.elser-2-elasticsearch
_inference/rerank/jina-reranker-v3
_inference/chat_completion/.gp-llm-v2-chat_completion
```

If these endpoints already exist in Elastic, no Jina or Anthropic key is needed in this app.

The Jina reranker follows Elastic's Jina integration:

```text
https://www.elastic.co/search-labs/integrations/jina
```

The LLM verdict agent calls Elastic's inference API:

```text
POST _inference/chat_completion/<LLM_INFERENCE_ID>
```

For Elastic Cloud, use the **Elasticsearch endpoint**, not the Kibana URL. The URL you shared is a Kibana URL:

```text
https://my-elasticsearch-project-a8bfb4.kb.us-east-1.aws.elastic.cloud/...
```

You need the Elasticsearch endpoint from the Elastic Cloud console. It usually looks like:

```text
https://<deployment-id>.es.us-east-1.aws.elastic.cloud:443
```

### Create Index and Pipeline

```bash
python -m src.create_index
```

If the index already exists and you want to recreate it before ingesting data:

```bash
python -m src.create_index --recreate
```

This setup command attempts to create:

```text
_ingest/pipeline/elser_sparse_pipeline
blr_claim_sources_v1
```

It does not create inference endpoints by default. Configure ELSER, Jina, and LLM endpoints in Elastic Inference Service, then set their IDs in `.env`.

### Generate Crawler Seed URLs

```bash
python -m src.crawler_config
```

This writes:

```text
config/crawler_seed_urls.txt
```

Use those URLs as the Open Crawler seeds with depth `2` and HTML content filtering.

For a quick local demo without configuring Open Crawler first, seed the index from the trusted sources:

```bash
python -m src.ingest_seed_sources --max-pages-per-source 5 --depth 1
```

This fetches HTML pages, extracts text, and indexes them through `elser_sparse_pipeline`, so ELSER vectors are generated during ingest.

To search the web for claim-related trusted pages first, index those pages, and then run retrieval:

```bash
python -m src.live_web_check "BMRCL is shutting Purple Line on Tuesday" --json
```

This is the full query-time demo path:

```text
claim -> trusted web search -> fetch pages -> ELSER pipeline ingest -> hybrid retrieval -> Jina rerank -> LLM verdict
```

### Check a Claim

```bash
python -m src.claim_check "BMRCL is shutting Purple Line on Tuesday"
```

### Run API

```bash
uvicorn src.api:app --reload
```

Then call:

```text
POST /check
{
  "claim": "BBMP banned plastic cups from cafes"
}
```
