import json
import unittest

from rag.pdf_loader import chunk_text, clean_text
from rag.prompts import load_prompt, render
from rag.retry import call_with_retries, is_retryable
from rag.schemas import ValidationFailure, parse_json, validate_answer
from tests.fakes import ApiError


class ChunkingTests(unittest.TestCase):
    def test_chunks_respect_size_and_overlap(self):
        text = " ".join(f"word{i}" for i in range(400))
        chunks = chunk_text(text, size=200, overlap=40)
        self.assertGreater(len(chunks), 5)
        self.assertTrue(all(len(c) <= 200 for c in chunks))
        self.assertTrue(any(w in chunks[1] for w in chunks[0].split()[-3:]))     # neighbours share words

    def test_short_text_is_one_chunk(self):
        self.assertEqual(chunk_text("Short text.", 500, 50), ["Short text."])

    def test_overlap_must_be_smaller_than_size(self):
        with self.assertRaises(ValueError):
            chunk_text("abc", size=10, overlap=10)

    def test_clean_text_fixes_pdf_line_breaks(self):
        self.assertEqual(clean_text("a multi-\nline  sentence\nhere"), "a multiline sentence here")


class SchemaTests(unittest.TestCase):
    GOOD = {"answer": "Yes.", "found": True, "citations": [2], "confidence": "high"}

    def test_valid_answer_passes(self):
        self.assertEqual(validate_answer(dict(self.GOOD), {2, 3})["citations"], [2])

    def test_code_fences_are_tolerated(self):
        self.assertEqual(parse_json("```json\n" + json.dumps(self.GOOD) + "\n```")["found"], True)

    def test_not_json_fails(self):
        with self.assertRaises(ValidationFailure):
            parse_json("Sure! The answer is 42")

    def test_missing_and_wrong_fields_are_all_reported(self):
        with self.assertRaises(ValidationFailure) as ctx:
            validate_answer({"answer": "", "found": "yes", "citations": "2", "confidence": "sure"}, {2})
        message = str(ctx.exception)
        for field in ("answer", "found", "citations", "confidence"):
            self.assertIn(field, message)

    def test_citation_outside_context_fails(self):
        with self.assertRaises(ValidationFailure):
            validate_answer({**self.GOOD, "citations": [9]}, {2})

    def test_found_needs_citations_and_refusal_must_have_none(self):
        with self.assertRaises(ValidationFailure):
            validate_answer({**self.GOOD, "citations": []}, {2})
        with self.assertRaises(ValidationFailure):
            validate_answer({**self.GOOD, "found": False}, {2})

    def test_extra_keys_fail(self):
        with self.assertRaises(ValidationFailure):
            validate_answer({**self.GOOD, "extra": 1}, {2})


class RetryTests(unittest.TestCase):
    def test_retries_temporary_errors_then_succeeds(self):
        calls, waits = [], []

        def flaky():
            calls.append(1)
            if len(calls) < 3:
                raise ApiError(503)
            return "ok"
        self.assertEqual(call_with_retries(flaky, max_attempts=4, sleep=waits.append), "ok")
        self.assertEqual(len(calls), 3)
        self.assertLess(waits[0], waits[1])                      # waiting grows

    def test_does_not_retry_a_wrong_key(self):
        calls = []

        def bad_key():
            calls.append(1)
            raise ApiError(401)
        with self.assertRaises(ApiError):
            call_with_retries(bad_key, max_attempts=4, sleep=lambda s: None)
        self.assertEqual(len(calls), 1)

    def test_gives_up_after_max_attempts(self):
        calls = []

        def always_busy():
            calls.append(1)
            raise ApiError(429)
        with self.assertRaises(ApiError):
            call_with_retries(always_busy, max_attempts=3, sleep=lambda s: None)
        self.assertEqual(len(calls), 3)

    def test_is_retryable(self):
        self.assertTrue(is_retryable(ApiError(503)))
        self.assertFalse(is_retryable(ApiError(404)))
        self.assertTrue(is_retryable(RuntimeError("Request timed out")))


class PromptTests(unittest.TestCase):
    def test_prompt_files_exist_and_have_placeholders(self):
        self.assertIn("<<CONTEXT>>", load_prompt("answer_prompt"))
        self.assertIn("<<QUESTION>>", load_prompt("answer_prompt"))
        self.assertIn("<<HISTORY>>", load_prompt("rewrite_prompt"))
        self.assertIn("<<ERROR>>", load_prompt("repair_prompt"))
        self.assertIn("<<ANSWER>>", load_prompt("judge_prompt"))

    def test_answer_prompt_contains_the_safety_rules(self):
        prompt = load_prompt("answer_prompt")
        self.assertIn("untrusted data", prompt)
        self.assertIn("ONLY the text inside", prompt)

    def test_render_is_single_pass(self):
        out = render("Q: <<QUESTION>> C: <<CONTEXT>>", QUESTION="<<CONTEXT>>", CONTEXT="doc")
        self.assertEqual(out, "Q: <<CONTEXT>> C: doc")          # text from the PDF is not re-scanned

    def test_render_reports_a_missing_value(self):
        with self.assertRaises(KeyError):
            render("<<NOPE>>")


if __name__ == "__main__":
    unittest.main()
