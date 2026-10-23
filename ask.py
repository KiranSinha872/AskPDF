"""
ask.py — Terminal loop for querying the Gas Turbine RAG chain.
Usage: python ask.py
Type "exit" to quit.
"""
from rag_chain import build_chain

def main():
    print("Building RAG chain...")
    chain = build_chain()
    print("Ready. Type your question (or 'exit' to quit).\n")

    while True:
        question = input("Q: ").strip()
        if not question:
            continue
        if question.lower() == "exit":
            print("Bye.")
            break

        result = chain.invoke(question)

        if not result["found"]:
            print("A: Not found\n")
        else:
            pages_str = ", ".join(str(p) for p in result["pages"])
            print(f"A: {result['answer']}")
            print(f"   Reference: Section \"{result['section']}\", Page {pages_str}")
            if result["quote"]:
                print(f"   Quote: \"{result['quote']}\"\n")
            else:
                print()


if __name__ == "__main__":
    main()
