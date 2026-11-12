"""Command-line chat with a PDF.

    python app.py --pdf data/matrix_basics.pdf                       # interactive chat
    python app.py --pdf data/matrix_basics.pdf -q "What is an identity matrix?"
    python app.py --list-models                                      # models your key can use
"""
import argparse
import json
import sys


def format_result(result):
    lines = [f"\nAnswer: {result.answer}"]
    if result.status == "ok":
        pages = ", ".join(str(p) for p in result.citations) or "-"
        lines.append(f"Pages: {pages}   Confidence: {result.confidence}")
    if result.standalone_question != result.question:
        lines.append(f"(searched for: {result.standalone_question})")
    if result.sources:
        lines.append("Sources:")
        for s in result.sources:
            lines.append(f"  - page {s.page} (distance {s.distance}): {s.snippet[:90]}...")
    lines.append(f"[status: {result.status}, tries: {result.attempts}, {result.latency_s}s]")
    return "\n".join(lines)


def main(argv=None, factory=None):
    parser = argparse.ArgumentParser(description="Chat with a PDF using retrieval + Gemini.")
    parser.add_argument("--pdf", default="data/matrix_basics.pdf", help="path to a text-based PDF")
    parser.add_argument("-q", "--question", help="ask one question and exit")
    parser.add_argument("--json", action="store_true", help="print the full result as JSON")
    parser.add_argument("--list-models", action="store_true", help="list models available to your API key")
    args = parser.parse_args(argv)

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    from rag.llm import explain_error
    try:
        if args.list_models:
            from rag.llm import GeminiClient, get_api_key
            for name in GeminiClient(get_api_key(), "unused").list_models():
                print(name)
            return 0
        if factory is None:
            from rag.factory import build_assistant as factory
        from rag.pipeline import ChatSession
        assistant = factory()
        stats = assistant.index_pdf(args.pdf)
    except Exception as error:
        print(f"Could not start: {explain_error(error)}", file=sys.stderr)
        return 1
    print(f"Indexed {args.pdf}: {stats['pages']} pages, {stats['chunks']} chunks.")

    session = ChatSession(assistant)
    if args.question:
        result = session.ask(args.question)
        print(json.dumps(result.to_dict(), indent=2) if args.json else format_result(result))
        return 0

    print("Ask questions about the PDF. Type 'quit' to stop, 'reset' to forget the conversation.")
    while True:
        try:
            question = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if question.lower() in ("quit", "exit"):
            break
        if question.lower() == "reset":
            session.reset()
            print("Conversation cleared.")
            continue
        result = session.ask(question)
        print(json.dumps(result.to_dict(), indent=2) if args.json else format_result(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
