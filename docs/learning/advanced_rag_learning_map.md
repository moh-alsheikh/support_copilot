To get to 70-80% you need 62-70 points. That's not 3x more tutorials, it's 2 more projects that force you into the 0% areas. Here is the exact map:

Rag 80 Roadmap

### The 14-week plan: 17% → 80%

**Your rule for scoring points:** For each bullet, 🟡 = you tried it once in code, ✅ = you evaluated it and can explain when it fails. Don't just watch.

#### Phase 0: Foundations - Week 1 (+5 pts → 22%)
Goal: Close terminology gaps.
- **Course:** DeepLearning.AI ChatGPT Prompt Engineering (1 day)
- **Book:** Hands-On Large Language Models by Jay Alammar, Chapters 1-4 - best visual explanation of embeddings
- **YouTube:** 3Blue1Brown - Attention + Embeddings

#### Phase 1: The Bottleneck - Query Understanding + Retrieval + Context - Weeks 2-4 (+20 pts → 42%)
This is where you get the most ROI. This alone makes you interview-ready.

- **Courses:**
  - DeepLearning.AI - Advanced Retrieval for AI with Chroma (free, 3 hours) - covers HyDE, multi-query, RRF
  - DeepLearning.AI - Building and Evaluating Advanced RAG (covers CRAG, Self-RAG, eval)
- **YouTube:**
  - James Briggs (Pinecone co-founder) - Advanced RAG series - 20 short videos, each one technique
  - LangChain - How to do RAPTOR + Contextual Compression
- **Papers to implement:** HyDE, RAG-Fusion, Contextual Retrieval by Anthropic (Dec 2024 blog + cookbook)
- **GitHub:** `lamhotsiagian/rag-techniques-playbook` - clone it and run techniques 5-15. This repo IS the 88 list in code.
- **Mini-project:** Add to your current project: query rewriting + 3x multi-query + HyDE + CRAG grader. Measure Recall@5 before/after.

**YouTube search:** "James Briggs Advanced RAG" and "Anthropic Contextual Retrieval"

#### Phase 2: Ingestion & Embedding Mastery - Weeks 5-6 (+10 pts → 52%)
- **Course:** LlamaIndex - Advanced Retrieval & Ingestion course
- **Book:** LLM Engineer's Handbook - Chapters 5-7 - best production chunking guide
- **Book:** Designing Data-Intensive Applications Ch. 2 & 12 - for HNSW, IVF, quantization
- **Papers:** ColBERTv2 (late interaction), Late Chunking (Jina AI), RAPTOR paper
- **Mini-project:** Rebuild ingestion: parent-child chunking + contextual retrieval (add LLM context to each chunk) + table-aware parsing. Compare fixed-size vs semantic vs contextual on your docs.

#### Phase 3: Agentic RAG - Weeks 7-9 (+15 pts → 68%) - You cross into senior territory here
- **Courses:**
  - LangChain Academy - Intro to LangGraph - free, this is THE agentic RAG course in 2025
  - DeepLearning.AI - AI Agentic Design Patterns with CrewAI
- **YouTube:** LangChain Agentic RAG tutorial, AI Engineer talks on Self-RAG
- **Papers:** Self-RAG (reflection tokens ISREL/ISSUP/ISUSE), ReAct, Adaptive-RAG
- **Mini-project 2 (new repo):** Build an agent that can route between: vector DB, web search (Tavily), and SQL. It should loop: think → search → check if enough → answer. This single project gives you 7/7 in Section 6.

#### Phase 4: Graph & Multimodal - Weeks 10-11 (+10 pts → 76%) - You hit your 70% target
- **Course:** Microsoft GraphRAG Accelerator Workshop on GitHub - official 2-hour workshop
- **Course:** Neo4j - GraphRAG Fundamentals (free)
- **YouTube:** "ColPali Visual Document Retrieval" tutorial - for multimodal PDFs
- **Papers:** GraphRAG (Microsoft 2024), LightRAG, HippoRAG
- **Mini-project 3:** Take a set of company docs with people/projects/relationships. Build GraphRAG vs vector RAG and show where graph wins on "how is X connected to Y?" queries.

#### Phase 5: Training + Production Polish - Weeks 12-14 (+8 pts → 84% stable)
- **Book:** Designing Machine Learning Systems by Chip Huyen - Chapters on eval, monitoring, cost
- **Course:** Cohere - Reranker Fine-tuning Guide + Pinecone - RAG in Production checklist
- **YouTube:** Chip Huyen - Production ML pitfalls
- **Papers:** RAFT - Retrieval Aware Fine-Tuning, LLMlingua for compression
- **Tasks:** Add to any project: semantic cache (GPTCache), access control filtering, eval dashboard with RAGAS, p95 latency logging, abstention logic. This is what you talk about in "can you ship production RAG?"

### Essential Resource List to Bookmark

**Books (in order):**
1.  Hands-On Large Language Models - Alammar
2.  LLM Engineer's Handbook - Iusztin & Labonne
3.  Designing Data-Intensive Applications - Kleppmann (skim for indexing)
4.  Designing Machine Learning Systems - Huyen (for Sections 11,12)

**Courses (all free except books):**
- DeepLearning.AI Advanced RAG stack (3 courses)
- LangChain Academy LangGraph
- Microsoft GraphRAG repo workshop

**YouTube Channels:**
- James Briggs
- LangChain
- AI Engineer
- Yannic Kilcher for paper deep dives

**GitHub Repos you must run, not just star:**
- `lamhotsiagian/rag-techniques-playbook` - do all 34
- `microsoft/graphrag`
- `deepset-ai/haystack` tutorials

### How to study for points

Don't watch then code. Code then watch. For each section:
1.  Implement naive version in your project
2.  Break it on purpose (e.g., ambiguous query, table question)
3.  Watch/read the technique that fixes that exact break
4.  Implement and add to eval set

If you follow Phase 1-4 and build Projects 2 and 3, you'll be at 65-76 points. Phase 5 pushes you past 80% and makes you production credible.

What type of docs are in your current project? I can tell you which 5 techniques from Phase 1 will give you +15 points this week.