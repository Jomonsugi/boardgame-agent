# Architecture

This document explains how the boardgame rules agent works, why each component exists, and how they connect. It's a reference for understanding the system and a starting point for iteration.

---

## System overview

```
User question
    │
    ▼
┌──────────┐     ┌──────────────────────────────────────────────────┐
│ Planner  │────▶│              ReAct Agent Loop                    │
│ (context │     │                                                  │
│  check)  │     │  System prompt with behavioral rules             │
└──────────┘     │       │                                          │
                 │       ▼                                          │
                 │  ┌─────────┐    ┌───────────────────────────┐    │
                 │  │  Agent  │◀──▶│         Tools             │    │
                 │  │  (LLM)  │    │  search_rulebook          │    │
                 │  └────┬────┘    │  lookup_glossary          │    │
                 │       │         │  view_page                │    │
                 │       │         │  search_web               │    │
                 │       │         │  get_past_answers         │    │
                 │       │         │  submit_answer            │    │
                 │       │         └───────────────────────────┘    │
                 │       ▼                                          │
                 │  submit_answer called → finalize                 │
                 └──────────────────────────────────────────────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │  QAWithCitations    │
              │  answer + citations │
              │  + web_sources      │
              └─────────────────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │  Streamlit UI       │
              │  Chat + PDF viewer  │
              │  with bbox highlights│
              └─────────────────────┘
```

---

## Core design principle

**The agent reasons iteratively when looking up a rule.** It starts by searching the most relevant source. After every search, it asks itself two questions:

1. "Can I fully answer now, with every claim grounded?"
2. "Is there anything in these results I don't understand?"

If yes to #1, it answers immediately. If yes to #2, it searches for the unknowns. This loop handles both simple questions (answered in one search) and complex multi-hop questions (requiring cross-references across documents) — without pre-classifying complexity.

This behavioral loop is entirely in the system prompt. It's not a separate planning system or graph structure. The LangGraph ReAct loop provides the mechanical loop; the prompt provides the reasoning intelligence.

---

## Document processing pipeline

### Extraction

```
PDF ──▶ Docling ──▶ per-page JSON with bounding boxes
                         │
                    (optional, on by default)
                         │
                    VLM enrichment: Qwen2.5-VL (3B, local, MPS)
                    describes each picture bbox visually
                    "Red starburst shape with the number 2."
                    (no interpretation of meaning — just shapes/colors)
```

**Why Docling?** It handles complex multi-column layouts, tables, and returns per-item bounding boxes with provenance. The bboxes are the foundation of the citation system — without them, we can't highlight specific text regions in the PDF viewer.

**Why VLM enrichment at extraction time?** Board game rulebooks communicate heavily through icons. Without VLM descriptions, picture bboxes have empty text and are invisible to search. The VLM prompt is deliberately minimal: "Describe exactly what you see: shapes, colors, numbers, and any text. Do not guess what it means or represents." Meaning is resolved later by the glossary, not here.

**Package: `docling`** — PDF parsing with per-item bounding boxes
**Package: `pymupdf` (fitz)** — PDF rendering, page cropping, bbox coordinate conversion

### Chunking

```
per-page JSON ──▶ chunk_by_sections() ──▶ section-level chunks
```

Each page's bboxes are grouped by heading labels (`section_header`, `title`). Tables become isolated chunks. Lone headings merge into the following section. Each chunk preserves `original_bbox_indices` mapping back to the page's bbox array — this is how citations trace from chunk → page → rendered highlight.

### Embedding and indexing

```
chunks ──▶ dense embedding (Ollama) + sparse embedding (SPLADE++)
       ──▶ Qdrant upsert with both vectors + full payload
```

**Why hybrid (dense + sparse)?** Dense embeddings (semantic) catch paraphrasing — "shield" matches "defense." Sparse embeddings (learned term weights) catch exact terminology — "Barkskin" matches "Barkskin." Neither alone is sufficient for rules text, which mixes precise game terms with natural language descriptions.

**Why RRF fusion?** Reciprocal Rank Fusion merges the ranked lists from dense and sparse search without requiring parameter tuning. It's Qdrant-native and runs server-side.

**Package: `ollama`** — local dense embeddings (`qwen3-embedding`, 4096-d)
**Package: `fastembed`** — SPLADE++ sparse embeddings (learned term weights)
**Package: `qdrant-client`** — vector database with hybrid search and RRF fusion

---

## Retrieval pipeline

```
query ──▶ embed (dense + sparse)
      ──▶ Qdrant prefetch (4×k candidates, RRF fusion)
      ──▶ cross-encoder re-ranking (Cohere or local FastEmbed)
      ──▶ top-k results formatted for LLM with bbox citation indices
```

### Re-ranking

**Why re-rank after RRF?** RRF fuses by rank position — it doesn't understand semantics. A cross-encoder scores each candidate against the query using cross-attention, catching subtle distinctions. Example: "Shield spell" vs "shield equipment" — both rank high in RRF, but the cross-encoder correctly ranks the spell higher when the query is about AC bonuses. Research shows 35-40% accuracy improvement from re-ranking (Pinecone benchmarks).

**Cohere Rerank** is the default because it's the highest quality and has a free tier (1,000 calls/month). Falls back to local **FastEmbed BGE-reranker-base** when no API key is set.

Qdrant does NOT support cross-encoder re-ranking natively. RRF is a fusion mechanism, not a re-ranker. The cross-encoder runs client-side on Qdrant's output. This is the standard two-stage retrieval pattern.

**Package: `cohere`** — Rerank API (free tier)
**Package: `fastembed`** — local cross-encoder fallback (BGE-reranker-base, 1GB)

### Formatted output to LLM

Retrieved chunks are formatted as:
```
=== DOCUMENT: The_Crew_Rules | PAGE 4 ===
[page text]

Bboxes (cite by index):
  [0] "First paragraph text..."
  [3] "Red starburst shape with the number 2."
```

The LLM sees document name, page number, and numbered bbox references. When it calls `submit_answer`, it includes these indices, which flow through to the UI as highlighted regions in the PDF viewer.

---

## Agent architecture

### LangGraph graph

```
planner ──▶ agent ◀──▶ tools ──▶ finalize ──▶ END
```

**Planner node**: Lightweight check — only detects when the answer is already in conversation context (follow-up questions, rephrased questions). On the first message, it's a no-op. All reasoning about what to search and how deep to go happens in the ReAct agent loop, not here.

**Agent node**: LLM with bound tools. Receives the system prompt (rebuilt fresh each call with current document list) and compressed message history. Makes tool calls until it calls `submit_answer`.

**Tools node**: Executes tool calls via LangGraph's ToolNode.

**Finalize node**: Extracts the JSON payload from `submit_answer`'s ToolMessage and writes it to `state["final_answer"]`. No LLM call.

**Recursion limit**: 15. The agent should answer within this. If not, the finalize node falls back to the agent's last text response.

**Message compression**: ToolMessages from previous turns are compressed to `"[retrieved N chars — already processed]"` to free context space. The agent only sees full results from the current turn.

**Package: `langgraph`** — stateful graph with ReAct loop, checkpointing, streaming
**Package: `langchain-core`** — message types, tool binding
**Package: `langchain-together/anthropic/openai`** — LLM provider integrations

### System prompt

The system prompt is the core intelligence of the system. It's rebuilt dynamically each call with:

- The current document list (names, tags, descriptions)
- Tool descriptions (conditional on what's available — glossary tool only appears when a glossary exists)
- A conversation-context skip marker (when planner detects a follow-up)
- Icon guidance (when glossary exists)

The critical section is "How to reason" — the introspection loop that teaches the agent to evaluate its own understanding after every search and keep going when gaps exist. This is what makes the agent cross-reference instead of answering from a single source.

### Tools

| Tool | Purpose | Produces citations? |
|------|---------|-------------------|
| `search_rulebook` | Hybrid search over indexed documents with tag filtering | Yes — doc_name, page_num, bbox_indices |
| `lookup_glossary` | Semantic search over the game's icon glossary | Yes — doc_name, page_num, bbox_indices from glossary entries |
| `view_page` | VLM analysis of a rendered page image | No — helps the agent understand what to search for next |
| `search_web` | Tavily web search restricted to configured domains | Yes — URL + finding (no bbox) |
| `get_past_answers` | Semantic search over accepted Q&A history | No — used for consistency, not citation |
| `submit_answer` | Formats the final answer with merged citations | N/A — this IS the output |

**Citation hierarchy**: `search_rulebook` and `lookup_glossary` are the primary citation sources. `view_page` is a comprehension aid — it helps the agent understand visual content, but the agent must then search for and cite the text-based rules. `search_web` provides URL citations but no bbox highlights.

---

## Icon glossary pipeline

### Why it exists

Board game rulebooks communicate through icons. A mission page might show numbered squares, colored starbursts, and symbol badges — all meaningful to the game, all invisible to text search. Without the glossary, the agent sees VLM descriptions like "dark square with number 1" but has no way to know this means "first task to complete in order."

The glossary bridges visual content to searchable game terms.

### Pipeline stages

```
All extracted pages for a game
    │
    ▼
1. Detect legend/reference pages (heuristic scoring)
   - Icon-reference documents (tagged or auto-detected from filename)
   - Pages with high icon density, short text labels, grid alignment, keywords
    │
    ▼
2. Build icon inventory
   - Crop all icon-sized picture bboxes from PDFs (area filtering)
   - Compute DHash (perceptual hash) for deduplication
   - Cluster by hash similarity (hamming distance ≤ 5)
    │
    ▼
3. Resolve meanings
   - Legend/reference pages: spatial proximity links each icon to adjacent text
     (centroid distance with directional bias — prefer below and right)
   - Hash matching: non-legend icons matched to legend entries by DHash
   - VLM resolution: unmatched icons ON LEGEND PAGES ONLY analyzed by a
     capable VLM with the full page image + already-known entries as context
   - Icons on non-legend pages that can't be hash-matched stay unresolved
    │
    ▼
4. Compute CLIP text embeddings for semantic search
    │
    ▼
5. Save glossary.json + invalidate agent cache
```

### Why spatial linking?

On legend pages, icons appear next to their text labels in a structured layout. The linking algorithm finds the nearest text bbox to each icon bbox using centroid-to-centroid distance, with a directional bias (prefer text below or to the right of the icon — the two most common legend layouts). This handles both "icon-above-caption" and "icon-left-of-label" patterns.

### Why DHash + CLIP?

**DHash (imagehash)**: "Are these the same icon?" Perceptual difference hashing is fast and robust to minor rendering variations. Used for deduplication and cross-page matching — if the same fire icon appears on 20 pages, they all cluster to one glossary entry.

**CLIP embeddings (open_clip)**: "What category does this icon belong to?" Enables semantic search in the `lookup_glossary` tool. When the agent searches for "resource icon" or "action symbol," CLIP finds semantically related entries even if the exact name doesn't match.

### Why VLM only on legend pages?

A VLM looking at a random rules page with icons would be guessing at meaning — the same failure mode as the extraction VLM. On a legend page, the meaning IS visually present (icon next to label). The VLM's job on a legend page is structured extraction from a known layout, not open-ended interpretation.

### Chunk enrichment

After the glossary is built, chunks can be re-indexed with icon meanings appended:
```
[original chunk text]

[Icons: order token 1 = first task to complete; order token 2 = second task to complete]
```

This makes icon meanings discoverable via normal text search, not just via the `lookup_glossary` tool.

**Package: `imagehash`** — perceptual hashing (DHash) for icon deduplication
**Package: `open-clip-torch`** — CLIP embeddings for semantic icon search

---

## Data storage

### SQLite (`games.db`)

| Table | Purpose |
|-------|---------|
| `games` | Registered games (game_id, game_name) |
| `documents` | Indexed docs per game (path, tag, description, VLM model, spreads) |
| `game_search_domains` | Per-game allowed web search domains |
| `qa_history` | Past Q&A pairs with embeddings for semantic lookup |

### Qdrant (local, file-based)

Single collection `rulebook_pages` with:
- Dense vectors (`qwen3-embedding`, cosine distance)
- Sparse vectors (SPLADE++, RRF-compatible)
- Payload: game_id, doc_name, doc_tag, page_num, text, bboxes, original_bbox_indices

Filtered by `game_id` on every query. Optionally filtered by `doc_tag`.

### File system

```
data/games/{game_id}/
├── docs/           # Stored document files (PDF, markdown)
├── extracted/      # Cached Docling extraction JSON (one per document)
└── glossary.json   # Icon glossary (built on demand)
```

Extraction is cached — Docling only runs once per document unless forced. VLM re-enrichment overwrites the cached JSON and triggers reindexing.

---

## UI architecture

### Streamlit layout

```
┌──────────┬────────────────────────┬────────────────────┐
│ Sidebar  │     Chat column        │   Document viewer  │
│          │                        │                    │
│ Game     │  User: question        │   PDF with bbox    │
│ selector │  Agent: answer         │   highlights       │
│          │    [citation chips]    │                    │
│ Documents│    [thumbs up/down]    │   or               │
│ list     │                        │   Markdown with    │
│          │  User: follow-up       │   text highlights  │
│ Glossary │  Agent: answer         │                    │
│          │    [citation chips]    │                    │
│ Upload   │                        │                    │
│          │  [chat input]          │                    │
│ Web      │                        │                    │
│ domains  │                        │                    │
└──────────┴────────────────────────┴────────────────────┘
```

Layout is adjustable (Chat / Equal / PDF presets). Citation clicks update the document viewer with highlighted bounding boxes.

### Agent caching

The compiled LangGraph agent is cached via `@st.cache_resource` keyed on `(game_id, model_name, enable_web_search)`. Building or rebuilding a glossary invalidates this cache so the `lookup_glossary` tool appears. Changing the model resets the conversation.

---

## Configuration reference

All configuration lives in `config.py`. Key settings:

| Setting | Default | Purpose |
|---------|---------|---------|
| `DEFAULT_MODEL` | Llama 3.3 70B (Together) | Agent LLM |
| `OLLAMA_EMBED_MODEL` | qwen3-embedding | Dense embeddings |
| `SPARSE_EMBED_MODEL` | SPLADE++ | Sparse embeddings |
| `RETRIEVAL_TOP_K` | 5 | Pages retrieved per query |
| `RERANK_PROVIDER` | cohere | Cross-encoder re-ranking |
| `VLM_DEFAULT_PRESET` | qwen (3B) | Local VLM for picture descriptions |
| `GLOSSARY_VLM_MODEL` | claude-sonnet-4-6 | VLM for glossary building |
| `PAGE_VISION_MODEL` | claude-sonnet-4-6 | VLM for page analysis tool |
| `ICON_AREA_MAX` | 5000 pts² | Max bbox area to consider as icon |
| `ICON_AREA_MIN` | 100 pts² | Min bbox area (filter noise) |
| `LEGEND_SCORE_THRESHOLD` | 0.4 | Pages scoring above this are legends |

---

## Hardware utilization

| Component | GPU (Apple MPS) | Notes |
|-----------|----------------|-------|
| Docling VLM enrichment | Yes | Hardcoded `AcceleratorDevice.MPS` |
| CLIP embeddings | Yes | Auto-detects MPS |
| Ollama dense embeddings | Yes | Ollama manages GPU internally |
| SPLADE++ sparse embeddings | No | FastEmbed, CPU — lightweight |
| DHash perceptual hashing | No | Pure numpy/PIL — fast on CPU |
| Cohere re-ranking | N/A | API call |
| LLM agent calls | N/A | API calls |

---

## Key design decisions and rationale

**Why ReAct introspection instead of a planner?** An early version classified questions as SIMPLE/COMPLEX upfront and generated retrieval plans. This was abandoned because question complexity can't be known before searching — a "simple" question about a shield spell may require cross-referencing three rule sections. The ReAct introspection loop discovers complexity at runtime, which analogues how humans actually look up rules.

**Why citations are mandatory?** The `submit_answer` tool requires citations. The system prompt says "you must call submit_answer to finish." This forces the agent to retrieve before answering — it can't hallucinate a rule because it has to point to where the rule is written. This is the most important quality control mechanism in the system.

**Why VLM descriptions are purely visual?** The extraction VLM prompt says "Do not guess what it means or represents." A 3B model guessing game meanings would hallucinate — the same icon means different things in different games. Visual descriptions are objective; meaning resolution requires cross-referencing, which is the glossary builder's job.

**Why the glossary builder's VLM is limited to legend pages?** A VLM analyzing a random rules page with icons would be guessing, just like the extraction VLM. On a legend page, the answer is visually present (icon next to label). Limiting VLM to legend pages eliminates a hallucination vector.

**Why `view_page` results are not citable?** VLM analysis helps the agent understand visual content, but it's not a source. "The VLM told me this icon means X" is not evidence — "the rulebook page 12 says this icon means X" is evidence. The agent must follow up VLM understanding with text retrieval to produce citable answers.

**Why Cohere over local re-ranking?** Cohere Rerank consistently outperforms open-source alternatives and has a free tier sufficient for a rules agent. The local FastEmbed fallback exists for offline use.

**Why not a knowledge graph?** A knowledge graph of game mechanics would help with multi-hop reasoning, but it requires game-specific ontology design. The current approach (glossary + introspective cross-referencing) handles multi-hop without game-specific structure. A knowledge graph may be worth exploring if the current approach hits limits on very complex rule interactions.

---

## Future considerations

**Corrective RAG (CRAG) — retrieval quality gating.** The Cohere re-ranker returns a `relevance_score` (0-1) per chunk that we currently discard after re-ordering. The CRAG pattern ([arxiv:2401.15884](https://arxiv.org/abs/2401.15884)) uses these scores to classify each retrieved chunk as Correct/Ambiguous/Incorrect before the LLM sees them. Low-scoring chunks are filtered out; if most chunks score poorly, the agent is told retrieval quality was low and should reformulate or try a different source. This would give the introspection loop a concrete signal instead of relying entirely on the model to judge content quality. Implementation requires calibrating score thresholds for the board game rules domain (~30-50 test queries per Cohere's guidance).

**RAGAS evaluation framework.** Reference-free metrics (context precision, context recall, faithfulness) for offline evaluation across a curated question set. Useful for measuring improvement across model and architecture changes. Target: faithfulness > 0.8.

**Task-specific re-ranker fine-tuning.** Generic re-rankers score for topical relevance, but board game rules questions need "answer utility" — a chunk may be topically relevant but not contain the specific rule. Fine-tuning a re-ranker on game rules data could improve precision, but requires collecting labeled examples.
