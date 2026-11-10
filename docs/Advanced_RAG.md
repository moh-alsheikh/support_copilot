## Advanced RAG

### Chat with meta.ai

https://meta.ai/prompt/0afcc92e-4fd8-4d39-a3a0-0495f3315955

### 1. Document Ingestion & Chunking - Advanced
Where most systems actually fail.

* **Fixed-size with overlap:** baseline 256-512 tokens, 10-20% overlap
* **Recursive / Structure-aware chunking:** split on headers, paragraphs, code AST, markdown, HTML
* **Semantic Chunking:** split where embedding similarity between sentences drops
* **Propositional Chunking:** convert document into atomic self-contained propositions before embedding
* **Late Chunking:** embed whole document first, then mean-pool token embeddings per chunk for better context awareness
* **Contextual Retrieval (Anthropic):** prepend LLM-generated context to each chunk to make it self-contained
* **Hierarchical / Parent-Child Chunking:** small child chunks for retrieval, large parent chunks for generation
* **Small-to-Big:** retrieve small, return surrounding big window
* **Metadata Enrichment:** auto-extract title, summary, entities, questions, hypothetical queries per chunk
* **Table / Layout-aware parsing:** preserve tables, charts, PDF layout with tools like Unstructured, LlamaParse, ColPali

### 2. Embedding & Representation
* **Bi-encoder vs Cross-encoder:** core distinction
* **Asymmetric embeddings:** different prompts for query vs document
* **Instruction-tuned embedders:** E5, BGE-M3, GTE, Nomic
* **Matryoshka Representation Learning (MRL):** truncatable embeddings for cost/performance trade-off
* **Late Interaction:** ColBERT, ColBERTv2, ColPali, ColQwen - multi-vector per token
* **Sparse-Dense Hybrid:** SPLADE, BM25 + dense
* **Multi-modal Embeddings:** text + image in same space
* **Code & Domain-specific Embedders:** Voyage-code, legal, medical, finance
* **Quantization:** Binary, int8, Product Quantization for large scale

### 3. Indexing & Storage Architecture
* **Vector Index Types:** HNSW, IVF, IVF-PQ, DiskANN
* **Hybrid Search:** combining BM25 lexical + vector with RRF - Reciprocal Rank Fusion or weighted sum
* **Re-ranking pipeline:** retrieve 100-200, rerank to 5-10 with cross-encoder
* **Hierarchical Index:** RAPTOR - recursive abstractive tree, summaries at different levels
* **Graph Index:** knowledge graph as index, not just vectors
* **Multi-index Routing:** route query to right index by domain, language, recency

### 4. Query Understanding & Transformation
* **Query Rewriting / Reformulation:** LLM rewrites user query for retrieval
* **Multi-Query:** generate 3-5 variations of same query
* **HyDE - Hypothetical Document Embeddings:** generate fake ideal answer and embed that
* **Step-Back Prompting:** ask broader high-level question to retrieve background
* **Query Decomposition:** break complex question into sub-questions
* **Query Expansion:** add synonyms, entities
* **Intent Classification / Routing:** factual vs analytical vs global vs local query

### 5. Retrieval Strategies - The core advanced types
* **Naive RAG:** single shot vector search
* **Advanced RAG:** hybrid + rerank
* **Modular RAG:** LEGO-like composable components
* **Adaptive RAG:** LLM decides if retrieval is needed at all
* **Self-RAG:** model emits reflection tokens - ISREL, ISSUP, ISUSE - to self-critique if docs are relevant and if answer is supported
* **Corrective RAG (CRAG):** retrieval evaluator grades docs, if low confidence fallback to web search or rewrite
* **Speculative RAG:** draft answer then verify with retrieval
* **Fusion RAG / RAG-Fusion:** RRF over multiple query variants
* **LongRAG / Long-Context RAG:** handling 100K+ context, dealing with lost-in-the-middle
* **MemoRAG:** build global memory / summary of whole corpus for global questions

### 6. Agentic RAG - The 2025 standard
* **Single-Agent RAG:** ReAct loop - think -> retrieve -> think -> retrieve
* **Multi-Agent RAG:** planner, retriever, synthesizer, verifier agents
* **Router-based Agentic RAG:** agent chooses tool - vector DB, SQL, API, web search
* **Tool-Using RAG:** retrieval as a function call
* **Self-Reflective / Iterative Retrieval:** agent judges answer and re-retrieves if not good
* **Planning:** query planning, multi-hop reasoning chains
* **Agent Memory:** short-term + long-term memory for RAG agents

### 7. Graph RAG & Structured RAG
* **GraphRAG (Microsoft):** extract entities + relationships into knowledge graph, community summaries for global queries
* **LightRAG, PathRAG, HippoRAG, HippoRAG 2:** more efficient graph variants
* **Tree-RAG / RAPTOR:** recursive summarization tree for abstractive retrieval
* **SQL / Text-to-SQL RAG:** retrieve from structured databases
* **KG + Vector Hybrid:** traverse graph + vector search
* **Entity Linking & Resolution:** connecting mentions across docs
* **Multi-hop Graph Traversal:** answering questions requiring 2-3+ hops

### 8. Context Handling & Generation
* **Context Compression:** RECOMP, LLMlingua, LongLLMLingua - compress chunks to fit more
* **Contextual Compression Retrieval:** only keep relevant sentences from each doc
* **Reordering / Lost-in-the-Middle mitigation:** place most relevant docs at start/end
* **Citation & Attribution:** force citations to sources
* **Answerability Check:** abstain if docs don't contain answer
* **Faithfulness / Groundedness:** reduce hallucination, verify every claim
* **Summarization-Augmented Generation:** summarize docs before generation
* **Chain-of-Verification:** generate then verify against retrieved docs

### 9. Multimodal & Specialized RAG
* **Multimodal RAG:** retrieve images, tables, PDFs, video frames
* **Visual Document Retrieval:** ColPali / ColQwen embedding page screenshots directly
* **Table RAG:** TAPAS, parsing and retrieving tables
* **Code RAG:** AST-aware chunking, repository-level retrieval
* **Voice / Audio RAG:** transcript + audio embeddings

### 10. Training & Fine-Tuning for RAG
* **Embedding Fine-tuning:** contrastive learning with InfoNCE, hard negative mining
* **Cross-encoder Fine-tuning:** fine-tune reranker on your domain
* **RAFT - Retrieval Aware Fine-Tuning:** train LLM to use retrieved context better
* **Context Distillation:** distill long context ability
* **Feedback training:** train on retrieval failures

### 11. Evaluation & Observability
* **Retrieval Metrics:** Recall@k, Precision@k, MRR, nDCG, Hit Rate
* **Generation Metrics:** Faithfulness, Relevancy, Context Relevance, Answer Correctness
* **RAG Triad:** Context Relevance, Groundedness, Answer Relevance
* **Frameworks:** RAGAS, ARES, TruLens, DeepEval
* **Failure Taxonomy:** retrieval miss, rerank miss, ignored context, contradicted source, wrongful refusal
* **Human vs LLM-as-Judge**

### 12. Production & Enterprise Advanced
* **Semantic Caching:** cache queries by embedding similarity
* **Query Result Caching & Invalidation**
* **Access Control & Security:** per-user filtering, cross-user retrieval leaks, private index
* **Freshness & Real-time Indexing:** incremental indexing, CDC
* **Scalability:** sharding, distributed vector DB
* **Cost Optimization:** tiered retrieval - cheap embed first, expensive rerank later
* **Personalized RAG:** user memory, history, profile-aware retrieval
* **Privacy-Preserving RAG:** federated, encrypted search

If you want to master advanced RAG in order, start here:
**Hybrid Search + Rerank -> Contextual Chunking -> Query Transformation (HyDE/Multi-Query) -> Self-RAG/CRAG -> Agentic RAG -> GraphRAG.**
