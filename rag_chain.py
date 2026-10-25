"""
rag_chain.py — Build the LCEL RAG chain for any document collection in ChromaDB using FastEmbed.

Grounding guarantees in postprocess():
  - Quote must appear (fuzzy-normalized) in the actual retrieved context text.
  - Every page number in `pages` must exist in the retrieved chunks' metadata.
  - If either check fails, return the canonical Not-found dict.
"""
import re
from typing import List, Dict

from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

import config


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------
class AnswerSchema(BaseModel):
    found: bool = Field(
        description="True if the answer is clearly and fully in the provided context, False otherwise"
    )
    answer: str = Field(description="Short direct answer, or empty string if not found")
    pages: List[int] = Field(
        description="1-based page numbers where the answer appears; empty list if not found"
    )
    section: str = Field(
        description="Section heading from the document; empty string if not found"
    )
    quote: str = Field(
        description="Verbatim supporting quote from the context, under 25 words; empty string if not found"
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
NOT_FOUND = {
    "answer": "Not found",
    "found": False,
    "pages": [],
    "section": "",
    "quote": "",
}


def _normalize(text: str) -> str:
    """Collapse whitespace and lower-case for loose substring matching."""
    return re.sub(r"\s+", " ", text).strip().lower()


def format_docs(docs) -> str:
    """Label each retrieved chunk with its 1-based page number."""
    parts = []
    for doc in docs:
        page = doc.metadata.get("page", "?")
        parts.append(f"[Page {page}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def postprocess(inputs: dict) -> dict:
    """
    Validate the LLM output against the retrieved context before returning.

    `inputs` has keys set by RunnablePassthrough.assign:
        "result"  – the AnswerSchema parsed by PydanticOutputParser
        "context" – the already-formatted context string (from format_docs)
        "chunks"  – the raw Doc list (for page-number verification)
    """
    parsed: AnswerSchema = inputs["result"]
    context_str: str = inputs["context"]
    chunks: list = inputs["chunks"]

    # Quick short-circuit
    if not parsed.found or not parsed.answer.strip():
        return NOT_FOUND

    norm_ctx = _normalize(context_str)

    # ── Check 1: quote must appear in the retrieved context ──────────────────
    quote = parsed.quote.strip()
    if quote:
        # Use the first 8 distinctive words of the quote for matching
        key_words = _normalize(quote).split()[:8]
        fragment = " ".join(key_words)
        if fragment and fragment not in norm_ctx:
            return NOT_FOUND
    else:
        # No quote given — treat as not-found
        return NOT_FOUND

    # ── Check 2: pages must exist in the retrieved chunks ───────────────────
    retrieved_pages = {
        int(doc.metadata["page"])
        for doc in chunks
        if "page" in doc.metadata
    }
    valid_pages = sorted(p for p in parsed.pages if p in retrieved_pages)
    if not valid_pages:
        return NOT_FOUND

    return {
        "answer": parsed.answer,
        "found": True,
        "pages": valid_pages,
        "section": parsed.section,
        "quote": quote,
    }


# ---------------------------------------------------------------------------
# Chain caching & builder
# ---------------------------------------------------------------------------
_chain_cache: Dict[str, any] = {}


def build_chain(doc_id: str = "default"):
    """
    Build or return cached LCEL RAG chain for a specific Chroma collection `doc_id`
    using the shared local FastEmbed model.
    """
    if doc_id in _chain_cache:
        return _chain_cache[doc_id]

    embeddings = config.get_embeddings()
    db = Chroma(
        collection_name=doc_id,
        persist_directory=config.DB_DIR,
        embedding_function=embeddings,
    )
    retriever = db.as_retriever(search_kwargs={"k": config.TOP_K})

    llm = ChatGoogleGenerativeAI(
        model=config.CHAT_MODEL,
        temperature=0,
        google_api_key=config.GOOGLE_API_KEY,
    )

    parser = PydanticOutputParser(pydantic_object=AnswerSchema)
    format_instructions = parser.get_format_instructions()

    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        partial_variables={"format_instructions": format_instructions},
        template="""You are a technical assistant for question answering over the document.

CONTEXT (retrieved passages — these are the ONLY facts you may use):
{context}

QUESTION: {question}

STRICT RULES:
1. Answer ONLY using information that is explicitly stated in the CONTEXT above. Do not use any outside knowledge.
2. Every statement in the answer must be directly supported by the quote. Do not add numbers, definitions, or facts that are not in the context.
3. If the context does not clearly and fully answer the question, set found=false.
4. If found=true: provide a short direct answer, the 1-based page number(s) taken from the [Page N] labels, the section heading from the document (or empty string if none is identifiable), and a verbatim quote under 25 words directly from the context.
5. If found=false: set answer, section, quote to empty strings and pages to [].

{format_instructions}
""",
    )

    def retrieve_and_format(question: str) -> dict:
        """Retrieve docs, format context string, return both."""
        docs = retriever.invoke(question)
        return {
            "chunks": docs,
            "context": format_docs(docs),
            "question": question,
        }

    chain = (
        RunnableLambda(retrieve_and_format)
        | RunnablePassthrough.assign(
            result=(
                RunnableLambda(lambda x: {"context": x["context"], "question": x["question"]})
                | prompt_template
                | llm
                | parser
            )
        )
        | RunnableLambda(postprocess)
    )

    _chain_cache[doc_id] = chain
    return chain
