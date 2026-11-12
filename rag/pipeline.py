"""The RAG pipeline: check question -> (rewrite follow-up) -> retrieve -> prompt -> LLM -> validate -> answer."""
import time
from dataclasses import asdict, dataclass, field

from .config import Config
from .llm import explain_error
from .pdf_loader import build_chunks, load_pdf_pages
from .prompts import PROMPTS_DIR, load_prompt, render
from .schemas import ValidationFailure, parse_json, validate_answer

NOT_FOUND = "I could not find that in the document."
INVALID_OUTPUT = "Sorry, I could not produce a valid answer. Please rephrase your question."


@dataclass
class Source:
    page: int
    chunk_id: str
    distance: float
    snippet: str
    text: str = ""


@dataclass
class AnswerResult:
    question: str
    standalone_question: str
    answer: str
    found: bool
    citations: list
    confidence: str
    status: str            # ok | no_relevant_context | invalid_question | invalid_output | llm_error
    sources: list = field(default_factory=list)
    attempts: int = 0      # how many LLM tries the answer needed (1 = valid the first time)
    latency_s: float = 0.0

    def to_dict(self):
        return asdict(self)


def format_history(history, turns=6):
    """Last few turns as plain text for the prompt."""
    if not history:
        return "(none)"
    lines = []
    for turn in history[-turns:]:
        lines.append(f"User: {turn['question']}")
        lines.append(f"Assistant: {turn['answer'][:300]}")
    return "\n".join(lines)


class RAGAssistant:
    def __init__(self, store, embedder, llm=None, config=None, prompts_dir=PROMPTS_DIR):
        self.store = store
        self.embedder = embedder
        self.llm = llm
        self.config = config or Config()
        self.prompts = {name: load_prompt(name, prompts_dir)
                        for name in ("answer_prompt", "rewrite_prompt", "repair_prompt")}

    # ---- indexing -------------------------------------------------------------------------
    def index_pdf(self, path):
        return self.index_pages(load_pdf_pages(path))

    def index_pages(self, pages):
        chunks = build_chunks(pages, self.config.chunk_size, self.config.chunk_overlap)
        self.store.reset()
        embeddings = self.embedder.embed([c.text for c in chunks])
        self.store.add(ids=[c.chunk_id for c in chunks], documents=[c.text for c in chunks],
                       metadatas=[{"page": c.page} for c in chunks], embeddings=embeddings)
        return {"pages": len(pages), "chunks": len(chunks)}

    # ---- steps ----------------------------------------------------------------------------
    def check_question(self, question):
        """Input guardrail. Returns a message if the question is not acceptable, else None."""
        text = (question or "").strip()
        if not text:
            return "Please type a question."
        if len(text) > self.config.max_question_chars:
            return f"Your question is too long (maximum {self.config.max_question_chars} characters)."
        return None

    def rewrite_question(self, question, history):
        """Turn a follow-up like 'Can you give an example?' into a standalone question."""
        question = question.strip()
        if not history or self.llm is None:
            return question
        try:
            prompt = render(self.prompts["rewrite_prompt"],
                            HISTORY=format_history(history, self.config.history_turns), QUESTION=question)
            rewritten = self.llm.generate(prompt, json_mode=False).strip().splitlines()[0].strip().strip('"')
            return rewritten if 3 <= len(rewritten) <= self.config.max_question_chars else question
        except Exception:
            return question            # if rewriting fails, search with the original question

    def retrieve(self, question):
        embedding = self.embedder.embed([question])[0]
        return self.store.query(embedding, self.config.top_k)

    def _generate_valid(self, prompt, allowed_pages):
        """Ask the LLM; if the JSON is invalid, ask again with the error (up to max_repairs times)."""
        current, last_error = prompt, ""
        for attempt in range(1, self.config.max_repairs + 2):
            raw = self.llm.generate(current, json_mode=True)
            try:
                return validate_answer(parse_json(raw), allowed_pages), attempt, ""
            except ValidationFailure as error:
                last_error = str(error)
                current = render(self.prompts["repair_prompt"],
                                 PROMPT=prompt, PREVIOUS=raw[:1500], ERROR=last_error)
        return None, self.config.max_repairs + 1, last_error

    # ---- main entry point -----------------------------------------------------------------
    def ask(self, question, history=None):
        start = time.time()

        def result(standalone, answer, found, citations, confidence, status, sources=(), attempts=0):
            return AnswerResult(question=question, standalone_question=standalone, answer=answer,
                                found=found, citations=citations, confidence=confidence, status=status,
                                sources=list(sources), attempts=attempts, latency_s=round(time.time() - start, 2))

        problem = self.check_question(question)
        if problem:
            return result(question, problem, False, [], "low", "invalid_question")

        standalone = self.rewrite_question(question, history)
        hits = self.retrieve(standalone)
        sources = [Source(h.page, h.id, round(h.distance, 3), h.text[:160], h.text) for h in hits]
        relevant = [h for h in hits if h.distance <= self.config.max_distance]
        if not relevant:                                  # nothing close enough: do not even call the LLM
            return result(standalone, NOT_FOUND, False, [], "low", "no_relevant_context", sources)

        context = "\n\n".join(f"[Page {h.page}] {h.text}" for h in relevant)
        prompt = render(self.prompts["answer_prompt"],
                        HISTORY=format_history(history, self.config.history_turns),
                        CONTEXT=context, QUESTION=standalone)
        try:
            parsed, attempts, _ = self._generate_valid(prompt, {h.page for h in relevant})
        except Exception as error:                        # network, key, quota, model name ...
            return result(standalone, f"The AI service could not answer: {explain_error(error)}",
                          False, [], "low", "llm_error", sources)
        if parsed is None:
            return result(standalone, INVALID_OUTPUT, False, [], "low", "invalid_output", sources, attempts)
        return result(standalone, parsed["answer"], parsed["found"], parsed["citations"],
                      parsed["confidence"], "ok", sources, attempts)


class ChatSession:
    """Keeps the conversation so follow-up questions work."""

    def __init__(self, assistant):
        self.assistant = assistant
        self.history = []

    def ask(self, question):
        result = self.assistant.ask(question, self.history)
        if result.status in ("ok", "no_relevant_context"):
            self.history.append({"question": question.strip(), "answer": result.answer})
        return result

    def reset(self):
        self.history = []
