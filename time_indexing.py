import os
import time
import psutil
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
import chromadb
import config

PDF_PATH = "data/GER-3620-P.pdf"

print("==================================================")
print(" PDF INDEXING TIMING & MEMORY BENCHMARK (FASTEMBED) ")
print("==================================================")

p = psutil.Process()
rss_start = p.memory_info().rss / (1024 * 1024)
print(f"Process Initial RSS: {rss_start:.2f} MB")

# Preload FastEmbed model
t0 = time.perf_counter()
embeddings = config.get_embeddings()
t_model_load = time.perf_counter() - t0
rss_after_model = p.memory_info().rss / (1024 * 1024)
print(f"Model Preload Time:  {t_model_load:.3f} s (RSS: {rss_after_model:.2f} MB)")

# 1. PDF Load
t0 = time.perf_counter()
loader = PyPDFLoader(PDF_PATH)
raw_docs = loader.load()
valid_docs = []
for doc in raw_docs:
    text = (doc.page_content or "").strip()
    if text:
        original_page = doc.metadata.get("page", 0)
        doc.metadata["page"] = int(original_page) + 1
        valid_docs.append(doc)
t_load = time.perf_counter() - t0
print(f"1. PDF Load Time:    {t_load:.3f} s ({len(raw_docs)} pages, {len(valid_docs)} with text)")

# 2. Splitting
t0 = time.perf_counter()
splitter = RecursiveCharacterTextSplitter(
    chunk_size=config.CHUNK_SIZE,
    chunk_overlap=config.CHUNK_OVERLAP,
)
chunks = splitter.split_documents(valid_docs)
t_split = time.perf_counter() - t0
print(f"2. Splitting Time:   {t_split:.3f} s ({len(chunks)} chunks)")

# 3. Embedding & Chroma Write
test_col = "timing_fastembed_benchmark"
t0 = time.perf_counter()
db = Chroma(
    collection_name=test_col,
    persist_directory=config.DB_DIR,
    embedding_function=embeddings,
)
db.add_documents(chunks)
t_embed_write = time.perf_counter() - t0
print(f"3. Embed & Write:    {t_embed_write:.3f} s (all {len(chunks)} chunks added)")

total_time = t_load + t_split + t_embed_write
rss_final = p.memory_info().rss / (1024 * 1024)

print("--------------------------------------------------")
print(f"TOTAL INDEXING TIME: {total_time:.3f} s")
print(f"FINAL PROCESS RSS:   {rss_final:.2f} MB")
print("==================================================")

# Cleanup
try:
    c_client = chromadb.PersistentClient(path=config.DB_DIR)
    c_client.delete_collection(test_col)
except Exception:
    pass
