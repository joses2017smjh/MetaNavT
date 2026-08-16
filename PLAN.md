# MetaNaviT Development Plan

## Phase 0: Foundation Upgrade — Best Open-Source Models & Tooling
**Goal:** Upgrade all AI models to the best available open-source alternatives, fix configuration issues, and establish a solid foundation.

### Tasks
- [x] Set up git config (Jose Sanchez / sanchej7@oregonstate.edu)
- [x] Create `.env` with upgraded model configuration
- [x] Upgrade LLM: `llama3.2:1b` → `qwen2.5:14b` (via Ollama) — best open-source 14B model
- [x] Upgrade Embedding: `all-MiniLM-L6-v2` → `BAAI/bge-large-en-v1.5` (1024 dims) — top MTEB retrieval
- [x] Fix embedding dimension mismatch (was 768 in config, model was 384)
- [x] Update `requirements.txt` with latest package versions (transformers, tokenizers, sentence-transformers, etc.)
- [x] Update `app/settings.py` — clean up model init, add proper defaults for new models
- [x] Update semantic chunker to use new embedding model
- [x] Update `.env.example` to reflect new defaults
- [x] Push initial Phase 0 commit

### Model Choices (Rationale)
| Component | Old | New | Why |
|-----------|-----|-----|-----|
| LLM | llama3.2:1b | qwen2.5:14b | Best open-source 14B; fits in RTX 8000 46GB; strong reasoning + coding |
| Embedding | all-MiniLM-L6-v2 (384d) | BAAI/bge-large-en-v1.5 (1024d) | Top MTEB retrieval scores; HuggingFace native |
| Tokenizer | tokenizers 0.21.0 | tokenizers (latest) | Bug fixes, speed improvements |
| Transformers | 4.48.3 | latest | New model support, optimizations |

### Hardware
- 2x Quadro RTX 8000 (46GB VRAM each, 92GB total)
- qwen2.5:14b needs ~10GB VRAM (Q4 quantized)
- bge-large-en-v1.5 needs ~1.3GB VRAM

---

## Phase 1: Enhanced Retrieval Pipeline
**Goal:** Dramatically improve retrieval quality by combining multiple search strategies and reranking.

### Tasks
- [x] Create `HybridRetriever` with RRF fusion (`app/engine/retriever.py`)
- [x] Integrate BM25 (ParadeDB) + dense vector (pgvector) retrieval
- [x] Add cross-encoder reranking with `BAAI/bge-reranker-v2-m3`
- [x] Wire hybrid retriever into query engine tool
- [x] Tune chunk sizes (512 tokens, 50 overlap) for `bge-large-en-v1.5`
- [x] Update `.env` / `.env.example` with reranker and chunk config
- [ ] Add query rewriting / HyDE (future improvement)

### Architecture
```
User Query
    │
    ├──► Vector Search (pgvector, top 10)
    │                                      ──► RRF Fusion ──► Reranker (bge-reranker-v2-m3) ──► Top 5 results
    └──► BM25 Search (ParadeDB, top 10)
```

## Phase 2: Multi-Modal & Advanced Agents
- Add vision capabilities using `llava` or `qwen2-vl` via Ollama
- Improve multi-agent handoff with better routing logic
- Add agent memory / context persistence across sessions

## Phase 3: Production Hardening
- Add proper error handling and retry logic throughout
- Implement health checks and monitoring
- Database migrations for schema changes
- Rate limiting and resource management

## Phase 4: User Experience
- Real-time file change detection (watchdog/inotify)
- Audio I/O with Whisper transcription
- Improved document chunking with semantic boundaries
- Human-in-the-loop approval UI improvements
