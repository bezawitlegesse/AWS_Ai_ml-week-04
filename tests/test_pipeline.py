import json
import unittest
from pathlib import Path

from eval.run_eval import build_report, run
from rag.config import Config
from rag.pdf_loader import load_pdf_pages
from rag.pipeline import INVALID_OUTPUT, NOT_FOUND, ChatSession, RAGAssistant
from rag.vector_store import InMemoryStore
from tests.fakes import ApiError, ExtractiveLLM, HashEmbedder, ScriptedLLM

ROOT = Path(__file__).resolve().parent.parent
PDF = ROOT / "data" / "matrix_basics.pdf"
GOOD_PAGE2 = json.dumps({"answer": "A square matrix with 1 on the diagonal and 0 elsewhere.",
                         "found": True, "citations": [2], "confidence": "high"})


def make_assistant(llm):
    # hashing embeddings give larger distances than real ones, so the threshold is relaxed for the tests
    assistant = RAGAssistant(InMemoryStore(), HashEmbedder(), llm, Config(max_distance=0.95))
    assistant.index_pdf(PDF)
    return assistant


class SamplePdfTests(unittest.TestCase):
    def test_sample_pdf_has_six_pages_with_text(self):
        pages = load_pdf_pages(PDF)
        self.assertEqual([p.number for p in pages], [1, 2, 3, 4, 5, 6])
        self.assertIn("identity matrix", pages[1].text.lower())


class RetrievalTests(unittest.TestCase):
    def test_identity_question_retrieves_page_2(self):
        hits = make_assistant(ScriptedLLM([])).retrieve("What is an identity matrix?")
        self.assertEqual(hits[0].page, 2)


class AskTests(unittest.TestCase):
    def test_valid_answer_with_citation(self):
        result = make_assistant(ScriptedLLM([GOOD_PAGE2])).ask("What is an identity matrix?")
        self.assertEqual((result.status, result.found, result.citations, result.attempts), ("ok", True, [2], 1))
        self.assertTrue(result.sources)

    def test_prompt_contains_context_with_page_labels_and_question(self):
        llm = ScriptedLLM([GOOD_PAGE2])
        make_assistant(llm).ask("What is an identity matrix?")
        self.assertIn("[Page 2]", llm.prompts[0])
        self.assertIn("Question: What is an identity matrix?", llm.prompts[0])

    def test_invalid_json_is_repaired(self):
        llm = ScriptedLLM(["Sure, here you go!", GOOD_PAGE2])
        result = make_assistant(llm).ask("What is an identity matrix?")
        self.assertEqual((result.status, result.attempts), ("ok", 2))
        self.assertIn("Problem:", llm.prompts[1])                 # the error message is sent back to the model

    def test_citation_to_a_page_not_retrieved_is_repaired(self):
        bad = json.dumps({"answer": "x", "found": True, "citations": [99], "confidence": "high"})
        result = make_assistant(ScriptedLLM([bad, GOOD_PAGE2])).ask("What is an identity matrix?")
        self.assertEqual((result.status, result.attempts), ("ok", 2))

    def test_gives_up_after_all_repairs_fail(self):
        result = make_assistant(ScriptedLLM(["no", "still no", "nope"])).ask("What is an identity matrix?")
        self.assertEqual((result.status, result.answer, result.found), ("invalid_output", INVALID_OUTPUT, False))
        self.assertEqual(result.attempts, 3)

    def test_question_with_no_relevant_context_skips_the_llm(self):
        llm = ScriptedLLM([])
        result = make_assistant(llm).ask("Who painted the Mona Lisa?")
        self.assertEqual((result.status, result.answer, result.found), ("no_relevant_context", NOT_FOUND, False))
        self.assertEqual(llm.prompts, [])

    def test_empty_and_too_long_questions_are_rejected(self):
        assistant = make_assistant(ScriptedLLM([]))
        self.assertEqual(assistant.ask("   ").status, "invalid_question")
        self.assertEqual(assistant.ask("x" * 501).status, "invalid_question")

    def test_api_failure_becomes_a_friendly_message(self):
        result = make_assistant(ScriptedLLM([ApiError(401, "bad key")])).ask("What is an identity matrix?")
        self.assertEqual(result.status, "llm_error")
        self.assertIn("API key", result.answer)

    def test_follow_up_is_rewritten_using_history(self):
        llm = ScriptedLLM(["What is an identity matrix used for?", GOOD_PAGE2])
        history = [{"question": "What is an identity matrix?", "answer": "A square matrix."}]
        result = make_assistant(llm).ask("What is it used for?", history)
        self.assertEqual(result.standalone_question, "What is an identity matrix used for?")
        self.assertIn("identity matrix used for", llm.prompts[1])

    def test_rewrite_failure_falls_back_to_the_original_question(self):
        llm = ScriptedLLM([RuntimeError("boom"), GOOD_PAGE2])
        history = [{"question": "What is an identity matrix?", "answer": "A square matrix."}]
        result = make_assistant(llm).ask("What is it used for?", history)
        self.assertEqual(result.standalone_question, "What is it used for?")

    def test_chat_session_remembers_only_successful_turns(self):
        session = ChatSession(make_assistant(ScriptedLLM([GOOD_PAGE2])))
        session.ask("")
        self.assertEqual(session.history, [])
        session.ask("What is an identity matrix?")
        self.assertEqual(len(session.history), 1)


class EvalRunnerTests(unittest.TestCase):
    def test_eval_runs_end_to_end_and_writes_a_report(self):
        golden = json.loads((ROOT / "eval" / "golden_set.json").read_text(encoding="utf-8"))
        assistant = make_assistant(ExtractiveLLM())
        rows, summary = run(assistant, golden, delay=0, progress=lambda m: None)
        self.assertEqual(summary["cases_total"], len(golden))
        self.assertIsNotNone(summary["overall_pass_rate"])
        report = build_report(rows, summary, assistant.config, retrieval_only=False)
        self.assertIn("## Metrics", report)
        self.assertIn("Failed cases", report)

    def test_retrieval_only_mode_needs_no_llm(self):
        golden = json.loads((ROOT / "eval" / "golden_set.json").read_text(encoding="utf-8"))
        assistant = make_assistant(None)
        rows, summary = run(assistant, golden, retrieval_only=True, progress=lambda m: None)
        self.assertEqual(summary["skipped"], 1)                   # the follow-up case needs the LLM
        self.assertGreater(summary["retrieval_hit_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()
