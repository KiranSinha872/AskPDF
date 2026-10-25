import os
import sys
import time
import subprocess
import json

MODELS = [
    ("MiniLM-L6", "sentence-transformers/all-MiniLM-L6-v2"),
    ("BGE-Small", "BAAI/bge-small-en-v1.5"),
]

THREADS_LIST = [2, 4, os.cpu_count() or 4]

# Worker code to run in a fresh process
WORKER_CODE = """
import os, sys, time, psutil, threading
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
import chromadb
from fastembed import TextEmbedding
from typing import List
from langchain_core.embeddings import Embeddings

model_name = sys.argv[1]
threads = int(sys.argv[2])
col_name = sys.argv[3]

p = psutil.Process()
peak_rss = [p.memory_info().rss]
stop_monitor = threading.Event()

def monitor():
    while not stop_monitor.is_set():
        try:
            rss = p.memory_info().rss
            if rss > peak_rss[0]:
                peak_rss[0] = rss
        except Exception:
            pass
        time.sleep(0.05)

t_thread = threading.Thread(target=monitor, daemon=True)
t_thread.start()

t0 = time.perf_counter()

# 1. Load PDF
loader = PyMuPDFLoader('data/GER-3620-P.pdf')
raw_docs = loader.load()
valid_docs = []
for doc in raw_docs:
    text = (doc.page_content or '').strip()
    if text:
        original_page = doc.metadata.get('page', 0)
        doc.metadata['page'] = int(original_page) + 1
        valid_docs.append(doc)
t_load = time.perf_counter() - t0

# 2. Split
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
chunks = splitter.split_documents(valid_docs)
t_split = time.perf_counter() - t0 - t_load

# 3. Embed & Write
class Wrapper(Embeddings):
    def __init__(self):
        self.model = TextEmbedding(model_name=model_name, threads=threads)
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [e.tolist() for e in self.model.passage_embed(texts, batch_size=32)]
    def embed_query(self, text: str) -> List[float]:
        return next(self.model.query_embed(text)).tolist()

w = Wrapper()
db = Chroma(collection_name=col_name, persist_directory='db', embedding_function=w)
SLICE_SIZE = 64
for i in range(0, len(chunks), SLICE_SIZE):
    db.add_documents(chunks[i : i + SLICE_SIZE])

t_total = time.perf_counter() - t0
stop_monitor.set()
t_thread.join(timeout=1.0)

# Cleanup collection
try:
    c = chromadb.PersistentClient(path='db')
    c.delete_collection(col_name)
except Exception:
    pass

res = {
    'load_s': t_load,
    'total_s': t_total,
    'peak_rss_mb': peak_rss[0] / (1024 * 1024)
}
import json
print('__RESULT__' + json.dumps(res))
"""

def run():
    print("================================================================================")
    print(" MEASURING MEMORY & TIME IN FRESH ISOLATED SUBPROCESSES")
    print(f" CPU Cores detected: {os.cpu_count()}")
    print("================================================================================")
    
    results = []
    
    for label, model_name in MODELS:
        # Deduplicate thread counts if cpu_count is 2 or 4
        seen_threads = []
        for th in THREADS_LIST:
            if th in seen_threads:
                continue
            seen_threads.append(th)
            th_label = f"all ({th})" if th == os.cpu_count() else str(th)
            col_name = f"fresh_{label.lower()}_th{th}"
            print(f"\nRunning {label} with threads={th_label} in fresh process...", flush=True)
            
            cmd = [sys.executable, "-c", WORKER_CODE, model_name, str(th), col_name]
            p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            
            output = p.stdout
            if "__RESULT__" in output:
                res_json = output.split("__RESULT__")[1].strip()
                data = json.loads(res_json)
                data["label"] = label
                data["threads"] = th_label
                results.append(data)
                print(f"-> Time: {data['total_s']:.2f}s | Peak RSS: {data['peak_rss_mb']:.1f} MB")
            else:
                print("Error in worker:", p.stderr[:400])

    print("\n\n================================================================================")
    print(" SUMMARY TABLE: FRESH PROCESS ISOLATION")
    print("================================================================================")
    print(f"{'Model':<12} | {'Threads':<10} | {'PDF Load (s)':<14} | {'Total Time (s)':<16} | {'Peak RSS (MB)':<14}")
    print("-" * 75)
    for r in results:
        print(f"{r['label']:<12} | {r['threads']:<10} | {r['load_s']:<14.2f} | {r['total_s']:<16.2f} | {r['peak_rss_mb']:<14.1f}")
    print("================================================================================")

if __name__ == "__main__":
    run()
