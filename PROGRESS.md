# Document Q&A / Gas Turbine Maintenance Assistant (RAG)
Goal: RAG over any PDF (including data/GER-3620-P.pdf) with Python + LangChain + ChromaDB + Gemini. Answer with section + page reference. Reply exactly "Not found" if the answer is not in the PDF. PydanticOutputParser before final output. Frontend for users. Deliver: GitHub repo, README (flow + answers + screenshots), 5 answers for the LMS form.

Questions: 1 three main types of inspection | 2 maintenance factor of two | 3 purchase price (Not found) | 4 melting point of titanium (Not found) | 5 recommended combustion-inspection interval in hours

Tech: config.py (UPLOAD_DIR=uploads, MAX_UPLOAD_MB=20, DB_DIR=db, CHAT_MODEL=gemini-3.5-flash-lite, EMBED_MODEL=BAAI/bge-small-en-v1.5, CHUNK_SIZE=1000, CHUNK_OVERLAP=150, TOP_K=8). Key in .env as GOOGLE_API_KEY (never commit or print). Chain: {"context": retriever | format_docs, "question": RunnablePassthrough()} | prompt | llm | PydanticOutputParser | RunnableLambda(postprocess). AnswerSchema(found, answer, pages, section, quote). Not-found output: {"answer":"Not found","found":false,"pages":[],"section":"","quote":""}. Pages are 1-based. Server: FastAPI POST /upload, POST /ask, GET /, GET /health, CORS on, uvicorn server:app --port 8000. Frontend: frontend/index.html, no frameworks.

RULES: Read this file first. Do only the next unchecked step. Do not redo finished steps or rewrite working files. Run commands yourself. After each step, tick it and add a line to the log. Write failures under Known issues.

CHECKLIST
- [x] Phase 0: env, config.py, PDF verified
- [x] A1 ingest.py run (db created)
- [x] A2 rag_chain.py (parser + postprocess)
- [x] A3 ask.py tested on Q1 and Q3
- [x] B1 server.py tested with curl
- [x] B2 frontend/index.html checked in browser
- [x] C1 run_questions.py, answers.md
- [x] C2 answers verified with pypdf
- [x] C3 screenshots q1..q5 in screenshots/
- [x] C4 README.md
- [x] C5 git commit (no .env) and push
- [x] D1 Generalise app: upload any PDF, dynamic doc_id Chroma collection, generalized prompt, frontend drag-and-drop & localStorage, verified with Python tests
- [x] D2 Faster indexing: FastEmbed local embeddings (BAAI/bge-small-en-v1.5), preloaded model, timing benchmark, and memory assessment
- [x] D3 Tune local embeddings: CPU core count threads, 64-chunk slicing, passage_embed/query_embed, PyMuPDFLoader (5x faster load), SHA-256 upload caching, and MiniLM-L6 vs BGE-small benchmark
- [x] D4 Finalise indexing: MiniLM default with EMBED_BACKEND env var, isolated process memory benchmarks across thread configurations, non-blocking upload with BackgroundTasks & GET /status/{doc_id}, 409 handling on /ask, frontend 1.5s polling with thin progress bar, and 100-page upload cap

LOG:
- Phase 0 done.
- A1 done: 60 pages loaded, 272 chunks created, db/ built with gemini-embedding-001 (3072-dim).
- A2 done: rag_chain.py with AnswerSchema, PydanticOutputParser, quote+page grounding in postprocess, stronger prompt.
- A3 done: tested Q1 (found, page 32), Q3 Not found, Q4 Not found, Q5 (found, page 33).
- B1+B2 done: server.py (FastAPI, CORS, lifespan chain build) + frontend/index.html verified. /health 200, Q1 found page 32, Q3 Not found, empty→400, GET / returns HTML.
- C1 done: run_questions.py executed across all 5 benchmark questions, generating answers.md.
- C2 done: verified with pypdf that all supporting quotes match verbatim on cited pages (Q1 on p. 25, Q2 on p. 9, Q5 on p. 33); Q3 and Q4 confirmed Not found; Q1 comprehensively captures both general maintenance inspection types (standby, running, disassembly) and Annular/Silo fleet types (Type A, B, C); TOP_K updated to 6 in config.py.
- C3 done: screenshots q1.png through q5.png captured from running frontend at localhost:8000 and saved in screenshots/.
- C4 done: comprehensive README.md written with architecture mermaid flow, setup guide, grounding mechanisms, benchmark answers table, embedded UI screenshots, and directory layout.
- C5 done: git repository initialized, files staged and committed with .env strictly ignored, and pushed to GitHub repository gas-turbine-rag.
- Step 9 done: restyled frontend to ChatGPT web app aesthetic (sidebar, clean chat thread, neutral colors, pulsing thinking indicator), recaptured screenshots q1..q5 (1440x900), updated README.md, and pushed to GitHub.
- D1 done: Generalized app to allow uploading and asking questions on any PDF up to 20MB. Updated config.py (UPLOAD_DIR="uploads", MAX_UPLOAD_MB=20), refactored ingest.py (ingest_pdf returning pages & chunks with scanned PDF detection and CLI support), generalized rag_chain.py (build_chain(doc_id) with collection-specific loading, prompt using "the document", and chain cache dict), updated server.py (POST /upload enforcing 20MB & .pdf format, POST /ask with doc_id and 404/400 validation), modernized frontend/index.html with drag-and-drop upload, sidebar file info, and localStorage session retention. Validated end-to-end with test_pipeline.py on GER-3620-P.pdf and a second solar PDF verifying cross-document isolation, grounding, and error states.
- D2 done: Replaced remote Gemini embeddings with local FastEmbed (BAAI/bge-small-en-v1.5). Removed batch sleep and 429 retry loops. Timed each stage: PDF load 4.07s, Splitting 0.006s. Embedding with FastEmbed eliminated 429 rate limit backoffs. Verified API end-to-end on GER-3620-P.pdf and test_solar.pdf with TOP_K=8. Measured server RSS: startup ~409MB, after full 60-page PDF indexing RSS reaches ~1450MB, which exceeds 400MB (Render 512MB RAM warning documented).
- D3 done: Tuned local embeddings pipeline: configured FastEmbed with batch_size=32 and threads=os.cpu_count(), added 64-chunk sliced Chroma writes keeping memory flat, ensured query_embed is used for queries and passage_embed for passages. Benchmarked both BAAI/bge-small-en-v1.5 (57.3s total) and sentence-transformers/all-MiniLM-L6-v2 (23.8s total). Replaced PyPDFLoader with PyMuPDFLoader (loaded 60 pages in 0.30s - 0.81s vs 4.07s). Confirmed layout and code retention on Binary-Trees-Complete-Course.pdf with diameter code query. Implemented SHA-256 upload caching (0.01s on duplicate uploads). Verified Q1 and Titanium questions on GER-3620-P.pdf.
- D4 done: Defaulted local embeddings to sentence-transformers/all-MiniLM-L6-v2 in config.py with EMBED_BACKEND env var support. Measured peak RSS and execution time across separate fresh processes: MiniLM-L6 completed in 16.61s (threads=4) and 17.92s (threads=2, lowest RSS 778 MB), easily beating the 40s target, while BGE-Small took 52-64s (>1050 MB RSS). Implemented non-blocking POST /upload returning {doc_id, status: "processing"} immediately and indexing via FastAPI BackgroundTasks with progress callbacks; added GET /status/{doc_id} returning batches_done, batches_total, pages, chunks, and error; made POST /ask return 409 'Still indexing' until document is ready; preserved instant SHA-256 cache hits (status: "ready" in 0.02s); enforced 100-page limit returning 400 Bad Request; updated frontend to poll /status every 1.5s with 'Indexing... X of Y' thin progress bar and 3-minute timeout guard. Verified 'give me the code of diameter' on Binary-Trees-Complete-Course.pdf matching O(N) solution on page 55, and confirmed GER-3620-P.pdf Q1 (found, pp. 32-34) and titanium question (Not found).

ISSUES:
- CHAT_MODEL changed from gemini-2.5-flash (404 for new users) to gemini-3.5-flash-lite (confirmed working) in config.py.
- Free-tier embedding quota: 100 RPM caused 429 backoff delays in remote Gemini embeddings; solved by switching to local FastEmbed (BAAI/bge-small-en-v1.5).
- Local FastEmbed memory: Process RSS peaks at ~1.0 - 1.45 GB during full 60-page indexing due to ONNX Runtime memory buffers on CPU. Exceeds 400 MB and will exceed a 512 MB Render instance limit.
- TOP_K updated to 8 for dense retrieval accuracy with bge-small.

LMS ANSWERS:
1. Maintenance inspection types are broadly classified as standby, running, and disassembly inspections (which progress from combustion to hot gas path to major inspections). For the GE Vernova Annular/Silo fleet, the three scheduled inspection types are Type A Visual Inspection, Type B Visual Inspection, and Type C Major Inspection (Section: Maintenance Inspections | Pages: 25, 32 | Quote: "Maintenance inspection types may be broadly classified as standby, running, and disassembly inspections.").
2. A maintenance factor of two indicates a maintenance interval that is half of the baseline interval (Section: Heavy-Duty Gas Turbine Operating and Maintenance Considerations | Page: 9 | Quote: "a maintenance factor of two would indicate a maintenance interval that is half of the baseline interval").
3. Not found
4. Not found
5. The recommended combustion-inspection interval in hours varies by turbine model and configuration, ranging from 8,000 to 32,000 hours (for example, 12,000 factored hours for 6B DLN and 32,000 factored hours for 7F DLN) (Section: Figure 36 . Baseline recommended inspection intervals: base load – natural gas fuel – dry | Page: 33 | Quote: "Combustion (DLN) Factored 12000/450").
