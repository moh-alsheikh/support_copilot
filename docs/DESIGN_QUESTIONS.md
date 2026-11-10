# Design Questions

Open architecture questions worth reasoning through before extending this project. Several are
answered in depth in [`THE_JOURNEY_TO_PRODUCTION.md`](THE_JOURNEY_TO_PRODUCTION.md) and
[`ARCHITECTURE_ASSESSMENT.md`](ARCHITECTURE_ASSESSMENT.md) — try answering from the code first.

- Which code is framework-specific, and which code would survive a framework change?
- Why does preparation precede LlamaIndex Documents?
- Incremental upserts skip unchanged documents, so which settings changes must still force a full rebuild, and why?
- What new operational problems would Pinecone or pgvector introduce?
- Which kind of user question would justify an agentic workflow instead of single-shot retrieval?
