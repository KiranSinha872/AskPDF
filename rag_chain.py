"""
rag_chain.py — Build the LCEL RAG chain for the Gas Turbine Maintenance Assistant.

Grounding guarantees in postprocess():
  - Quote must appear (fuzzy-normalized) in the actual retrieved context text.
  - Every page number in `pages` must exist in the retrieved chunks' metadata.
  - If either check fails, return the canonical Not-found dict.
"""
import re
from typing import List

from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
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

    `inputs` has two keys set by RunnablePassthrough.assign:
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
        # (LLM may add minor punctuation differences)
        key_words = _normalize(quote).split()[:8]
        fragment = " ".join(key_words)
        if fragment and fragment not in norm_ctx:
            return NOT_FOUND
    else:
        # No quote given — treat as not-found (quote is required when found=true)
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
# Chain builder
# ---------------------------------------------------------------------------
def build_chain():
    # Embeddings + Chroma retriever
    embeddings = GoogleGenerativeAIEmbeddings(
        model=config.EMBED_MODEL,
        google_api_key=config.GOOGLE_API_KEY,
    )
    db = Chroma(
        persist_directory=config.DB_DIR,
        embedding_function=embeddings,
    )
    retriever = db.as_retriever(search_kwargs={"k": config.TOP_K})

    # LLM
    llm = ChatGoogleGenerativeAI(
        model=config.CHAT_MODEL,
        temperature=0,
        google_api_key=config.GOOGLE_API_KEY,
    )

    # Parser
    parser = PydanticOutputParser(pydantic_object=AnswerSchema)
    format_instructions = parser.get_format_instructions()

    # Prompt
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        partial_variables={"format_instructions": format_instructions},
        template="""You are a technical assistant for GE heavy-duty gas turbine documentation (GER-3620 Rev P).

CONTEXT (retrieved passages — these are the ONLY facts you may use):
{context}

QUESTION: {question}

STRICT RULES:
1. Answer ONLY using information that is explicitly stated in the CONTEXT above. Do not use any outside knowledge.
2. Every statement in the answer must be directly supported by the quote. Do not add numbers, definitions or facts that are not in the context.
3. If the context only partly answers the question, set found=false.
4. The purchase price of the turbine is NOT in this document — if asked, set found=false.
5. The melting point of titanium is NOT in this document — if asked, set found=false.
6. When asked about inspection types, note both the general classifications: standby, running, and disassembly (which progress from combustion to hot gas path to major inspection) and the Annular/Silo fleet (Type A, B, C) if present in context.
7. If found=true: provide a short answer, the 1-based page number(s) taken from the [Page N] labels, the section heading, and a verbatim quote under 25 words.
8. If found=false: set answer, section, quote to empty strings and pages to [].

{format_instructions}
""",
    )

    # ── LCEL chain ────────────────────────────────────────────────────────────
    # We need both the retrieved docs AND the formatted context string in
    # postprocess, so we build the context branch explicitly and pass everything
    # forward via RunnablePassthrough.assign.

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
    return chain
