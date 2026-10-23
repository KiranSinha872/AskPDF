"""
run_questions.py — Ask the 5 target questions to the RAG chain and save answers.md.
"""
import sys
import rag_chain

QUESTIONS = [
    "What are the three main types of inspection?",
    "What does a maintenance factor of two mean?",
    "What is the purchase price of this gas turbine?",
    "What is the melting point of titanium?",
    "What is the recommended combustion-inspection interval in hours?",
]

def main():
    print("Building RAG chain...")
    chain = rag_chain.build_chain()
    
    results = []
    print("\nRunning questions through chain:")
    for idx, q in enumerate(QUESTIONS, 1):
        print(f"\n[{idx}/5] Question: {q}")
        res = chain.invoke(q)
        print(f"Result: {res}")
        results.append((q, res))
        
    # Write answers.md
    with open("answers.md", "w") as f:
        f.write("# Target Questions & Answers\n\n")
        f.write("| Question | Answer | Section | Page | Quote |\n")
        f.write("|---|---|---|---|---|\n")
        for q, res in results:
            pages_str = ", ".join(str(p) for p in res.get("pages", [])) if res.get("pages") else "-"
            ans_str = res.get("answer", "").replace("\n", " ").replace("|", "\\|")
            sec_str = res.get("section", "").replace("\n", " ").replace("|", "\\|") or "-"
            quote_str = res.get("quote", "").replace("\n", " ").replace("|", "\\|") or "-"
            f.write(f"| {q} | {ans_str} | {sec_str} | {pages_str} | {quote_str} |\n")
            
    print("\nSaved answers to answers.md")

if __name__ == "__main__":
    main()
