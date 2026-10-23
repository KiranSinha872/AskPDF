# Gas Turbine Maintenance Assistant (RAG)
Goal: RAG over data/GER-3620-P.pdf (GE GER-3620 Rev P, 60 pp) with Python + LangChain + ChromaDB + Gemini. Answer with section + page reference. Reply exactly "Not found" if the answer is not in the PDF. PydanticOutputParser before final output. Frontend for users. Deliver: GitHub repo, README (flow + answers + screenshots), 5 answers for the LMS form.

Questions: 1 three main types of inspection | 2 maintenance factor of two | 3 purchase price (Not found) | 4 melting point of titanium (Not found) | 5 recommended combustion-inspection interval in hours

Tech: config.py (PDF_PATH, DB_DIR=db, CHAT_MODEL=gemini-2.5-flash, EMBED_MODEL=models/gemini-embedding-001, CHUNK_SIZE=1000, CHUNK_OVERLAP=150, TOP_K=4). Key in .env as GOOGLE_API_KEY (never commit or print). Chain: {"context": retriever | format_docs, "question": RunnablePassthrough()} | prompt | llm | PydanticOutputParser | RunnableLambda(postprocess). AnswerSchema(found, answer, pages, section, quote). Not-found output: {"answer":"Not found","found":false,"pages":[],"section":"","quote":""}. Pages are 1-based. Server: FastAPI POST /ask, GET /, GET /health, CORS on, uvicorn server:app --port 8000. Frontend: frontend/index.html, no frameworks.

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

ISSUES:
- CHAT_MODEL changed from gemini-2.5-flash (404 for new users) to gemini-3.5-flash-lite (confirmed working) in config.py.
- Free-tier embedding quota: 100 RPM. ingest.py sleeps 65s between batches of 50.
- TOP_K increased from 4 to 6 so that Page 9 (maintenance factor definition) and Page 25 (standby/running/disassembly inspection types) are retrieved alongside Page 32.

LMS ANSWERS:
1. Maintenance inspection types are broadly classified as standby, running, and disassembly inspections (which progress from combustion to hot gas path to major inspections). For the GE Vernova Annular/Silo fleet, the three scheduled inspection types are Type A Visual Inspection, Type B Visual Inspection, and Type C Major Inspection (Section: Maintenance Inspections | Pages: 25, 32 | Quote: "Maintenance inspection types may be broadly classified as standby, running, and disassembly inspections.").
2. A maintenance factor of two indicates a maintenance interval that is half of the baseline interval (Section: Heavy-Duty Gas Turbine Operating and Maintenance Considerations | Page: 9 | Quote: "a maintenance factor of two would indicate a maintenance interval that is half of the baseline interval").
3. Not found
4. Not found
5. The recommended combustion-inspection interval in hours varies by turbine model and configuration, ranging from 8,000 to 32,000 hours (for example, 12,000 factored hours for 6B DLN and 32,000 factored hours for 7F DLN) (Section: Figure 36 . Baseline recommended inspection intervals: base load – natural gas fuel – dry | Page: 33 | Quote: "Combustion (DLN) Factored 12000/450").
