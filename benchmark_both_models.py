import os
import time
import psutil
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
import chromadb
from fastembed import TextEmbedding
from typing import List
from langchain_core.embeddings import Embeddings

PDF_PATH = "data/GER-3620-P.pdf"
CPU_CORES = os.cpu_count() or 4

class FastEmbedBenchWrapper(Embeddings):
    def __init__(self, model_name: str, threads: int = CPU_CORES, batch_size: int = 32):
        self.batch_size = batch_size
        self.model = TextEmbedding(model_name=model_name, threads=threads)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [e.tolist() for e in self.model.passage_embed(texts, batch_size=self.batch_size)]

    def embed_query(self, text: str) -> List[float]:
        return next(self.model.query_embed(text)).tolist()

def benchmark_model(model_name: str, label: str):
    print(f"\n==========================================")
    print(f" BENCHMARK: {label} ({model_name})")
    print(f"==========================================")
    p = psutil.Process()
    
    # Preload model
    t0 = time.perf_counter()
    wrapper = FastEmbedBenchWrapper(model_name=model_name)
    t_model_load = time.perf_counter() - t0
    rss_after_load = p.memory_info().rss / (1024 * 1024)
    print(f"Model Preload Time: {t_model_load:.3f} s (RSS: {rss_after_load:.1f} MB)")
    
    # 1. PDF Load with PyMuPDFLoader
    t0 = time.perf_counter()
    loader = PyMuPDFLoader(PDF_PATH)
    raw_docs = loader.load()
    valid_docs = []
    for doc in raw_docs:
        text = (doc.page_content or "").strip()
        if text:
            original_page = doc.metadata.get("page", 0)
            doc.metadata["page"] = int(original_page) + 1
            valid_docs.append(doc)
    t_load = time.perf_counter() - t0
    print(f"1. PDF Load (PyMuPDF): {t_load:.3f} s ({len(raw_docs)} pages)")

    # 2. Splitting
    t0 = time.perf_counter()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(valid_docs)
    t_split = time.perf_counter() - t0
    print(f"2. Splitting:          {t_split:.3f} s ({len(chunks)} chunks)")

    # 3. Sliced indexing in blocks of 64
    test_col = f"bench_{label.lower().replace(' ', '_')}"
    db = Chroma(collection_name=test_col, persist_directory="db", embedding_function=wrapper)
    
    SLICE_SIZE = 64
    peak_rss = p.memory_info().rss / (1024 * 1024)
    t0 = time.perf_counter()
    for i in range(0, len(chunks), SLICE_SIZE):
        slice_chunks = chunks[i : i + SLICE_SIZE]
        db.add_documents(slice_chunks)
        curr_rss = p.memory_info().rss / (1024 * 1024)
        if curr_rss > peak_rss:
            peak_rss = curr_rss
    t_embed_write = time.perf_counter() - t0
    print(f"3. Sliced Embed+Write: {t_embed_write:.3f} s")
    
    total_time = t_load + t_split + t_embed_write
    print(f"Total Time:            {total_time:.3f} s")
    print(f"Peak Process RSS:      {peak_rss:.1f} MB")
    
    # Cleanup Chroma collection
    c = chromadb.PersistentClient(path="db")
    c.delete_collection(test_col)
    
    return {
        "model": model_name,
        "load_s": t_load,
        "embed_s": t_embed_write,
        "total_s": total_time,
        "peak_rss_mb": peak_rss
    }

if __name__ == "__main__":
    res_bge = benchmark_model("BAAI/bge-small-en-v1.5", "BGE-Small")
    res_minilm = benchmark_model("sentence-transformers/all-MiniLM-L6-v2", "MiniLM-L6")
    
    print("\n\n========================================================")
    print(" SUMMARY COMPARISON TABLE")
    print("========================================================")
    print(f"{'Metric':<25} | {'BGE-Small-v1.5':<18} | {'all-MiniLM-L6-v2':<18}")
    print("-" * 68)
    print(f"{'PDF Load (s)':<25} | {res_bge['load_s']:<18.3f} | {res_minilm['load_s']:<18.3f}")
    print(f"{'Embed + Write (s)':<25} | {res_bge['embed_s']:<18.3f} | {res_minilm['embed_s']:<18.3f}")
    print(f"{'Total Indexing (s)':<25} | {res_bge['total_s']:<18.3f} | {res_minilm['total_s']:<18.3f}")
    print(f"{'Peak RSS (MB)':<25} | {res_bge['peak_rss_mb']:<18.1f} | {res_minilm['peak_rss_mb']:<18.1f}")
    print("========================================================")
