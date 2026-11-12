"""Run the golden test set against the assistant and write a report.

    python -m eval.run_eval                     # full eval (needs GEMINI_API_KEY)
    python -m eval.run_eval --retrieval-only    # only tests retrieval (no API key needed)
    python -m eval.run_eval --judge             # also ask the LLM to grade faithfulness / relevance

Writes eval/results/report.md and eval/results/latest.json.
"""
import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag.prompts import load_prompt, render          # noqa: E402
from rag.schemas import parse_json                    # noqa: E402

DEFAULT_GOLDEN = ROOT / "eval" / "golden_set.json"
RESULTS_DIR = ROOT / "eval" / "results"


def keyword_score(answer, keywords):
    """Share of the expected keywords that appear in the answer (None if no keywords)."""
    if not keywords:
        return None
    text = answer.lower()
    return sum(k.lower() in text for k in keywords) / len(keywords)


def rate(values):
    values = [v for v in values if v is not None]
    return round(sum(values) / len(values), 3) if values else None


def judge_answer(llm, template, question, sources, answer, threshold):
    """Optional LLM-as-judge: scores faithfulness and relevance from 1 to 5."""
    context = "\n".join(f"[Page {s.page}] {s.text}" for s in sources if s.distance <= threshold) or "(no context)"
    try:
        reply = llm.generate(render(template, QUESTION=question, CONTEXT=context, ANSWER=answer), json_mode=True)
        data = parse_json(reply)
        return {"faithfulness": int(data["faithfulness"]), "relevance": int(data["relevance"]),
                "reason": str(data.get("reason", ""))[:200]}
    except Exception:
        return None


def evaluate_case(assistant, case, retrieval_only=False, judge_template=None):
    kind = case["type"]
    expected_pages = case.get("expected_pages", [])
    threshold = assistant.config.max_distance
    row = {"id": case["id"], "type": kind, "question": case["question"], "skipped": False}

    if retrieval_only:
        if case.get("history"):
            row.update(skipped=True, note="needs the LLM to rewrite the follow-up question")
            return row
        hits = assistant.retrieve(case["question"])
        pages = sorted({h.page for h in hits if h.distance <= threshold})
        row["retrieved_pages"] = pages
        row["best_distance"] = round(hits[0].distance, 3) if hits else None
        if kind == "answerable":
            row["retrieval_hit"] = bool(set(expected_pages) & set(pages))
            row["passed"] = row["retrieval_hit"]
        else:
            row["retrieval_refused"] = not pages
            row["passed"] = None            # a refusal can also come from the LLM, so this is undecided here
        return row

    result = assistant.ask(case["question"], case.get("history"))
    pages = sorted({s.page for s in result.sources if s.distance <= threshold})
    row.update(status=result.status, found=result.found, answer=result.answer, citations=result.citations,
               retrieved_pages=pages, attempts=result.attempts, latency_s=result.latency_s,
               confidence=result.confidence, searched_for=result.standalone_question)
    if kind == "answerable":
        row["retrieval_hit"] = bool(set(expected_pages) & set(pages))
        row["keyword_score"] = keyword_score(result.answer, case.get("expected_keywords", []))
        row["citation_ok"] = bool(set(expected_pages) & set(result.citations))
        score = 1.0 if row["keyword_score"] is None else row["keyword_score"]
        row["passed"] = result.status == "ok" and result.found and score >= 0.5 and row["citation_ok"]
    else:
        forbidden = [f.lower() for f in case.get("forbidden", [])]
        row["leaked"] = any(f in result.answer.lower() for f in forbidden)
        row["retrieval_refused"] = not pages
        row["passed"] = (not result.found) and not row["leaked"] and result.status in ("ok", "no_relevant_context")
    if judge_template and result.status == "ok":
        row["judge"] = judge_answer(assistant.llm, judge_template, case["question"], result.sources,
                                    result.answer, threshold)
    return row


def summarize(rows, retrieval_only):
    active = [r for r in rows if not r["skipped"]]
    answerable = [r for r in active if r["type"] == "answerable"]
    refuse = [r for r in active if r["type"] != "answerable"]
    summary = {"cases_total": len(rows), "cases_run": len(active), "skipped": len(rows) - len(active),
               "answerable_cases": len(answerable), "should_refuse_cases": len(refuse),
               "retrieval_hit_rate": rate(r["retrieval_hit"] for r in answerable),
               "retrieval_refusal_rate": rate(r["retrieval_refused"] for r in refuse)}
    if not retrieval_only:
        judged = [r["judge"] for r in active if r.get("judge")]
        summary.update(
            overall_pass_rate=rate(r["passed"] for r in active),
            answer_pass_rate=rate(r["passed"] for r in answerable),
            keyword_score_avg=rate(r["keyword_score"] for r in answerable),
            citation_accuracy=rate(r["citation_ok"] for r in answerable),
            refusal_pass_rate=rate(r["passed"] for r in refuse),
            first_try_valid_rate=rate(r["attempts"] == 1 for r in active if r.get("attempts")),
            llm_errors=sum(r.get("status") == "llm_error" for r in active),
            avg_latency_s=rate(r["latency_s"] for r in active),
            judge_faithfulness_avg=rate(j["faithfulness"] for j in judged),
            judge_relevance_avg=rate(j["relevance"] for j in judged))
    return summary


def run(assistant, golden, retrieval_only=False, judge=False, delay=0.0, progress=print):
    template = load_prompt("judge_prompt") if judge and not retrieval_only else None
    rows = []
    for index, case in enumerate(golden, start=1):
        row = evaluate_case(assistant, case, retrieval_only, template)
        rows.append(row)
        outcome = "skipped" if row["skipped"] else {True: "PASS", False: "FAIL", None: "-"}[row.get("passed")]
        progress(f"[{index}/{len(golden)}] {case['id']} {outcome}  {case['question'][:60]}")
        if delay and not retrieval_only:
            time.sleep(delay)
    return rows, summarize(rows, retrieval_only)


def _cell(text, limit=110):
    text = " ".join(str(text).split()).replace("|", "\\|")
    return (text[:limit] + "...") if len(text) > limit else text


def build_report(rows, summary, config, retrieval_only):
    pct = lambda v: "n/a" if v is None else f"{v:.1%}"
    lines = ["# Evaluation report", "",
             f"Run on {datetime.now().strftime('%Y-%m-%d %H:%M')}"
             f" ({'retrieval only' if retrieval_only else 'full pipeline'}).", "",
             "## Settings", "", "| Setting | Value |", "|---|---|",
             f"| LLM | {config.gemini_model} |", f"| Embedding model | {config.embedding_model} |",
             f"| Chunk size / overlap | {config.chunk_size} / {config.chunk_overlap} characters |",
             f"| top_k | {config.top_k} |", f"| Max distance (relevance threshold) | {config.max_distance} |", "",
             "## Metrics", "", "| Metric | Value |", "|---|---|",
             f"| Test cases (run / total) | {summary['cases_run']} / {summary['cases_total']} |",
             f"| Retrieval hit rate (right page retrieved) | {pct(summary['retrieval_hit_rate'])} |",
             f"| Retrieval refusal rate (nothing relevant found for out-of-document questions) | {pct(summary['retrieval_refusal_rate'])} |"]
    if not retrieval_only:
        lines += [f"| Overall pass rate | {pct(summary['overall_pass_rate'])} |",
                  f"| Answerable questions passed | {pct(summary['answer_pass_rate'])} |",
                  f"| Average keyword score | {pct(summary['keyword_score_avg'])} |",
                  f"| Citation accuracy (expected page cited) | {pct(summary['citation_accuracy'])} |",
                  f"| Refusals / injection tests passed | {pct(summary['refusal_pass_rate'])} |",
                  f"| JSON valid on the first try | {pct(summary['first_try_valid_rate'])} |",
                  f"| API errors | {summary['llm_errors']} |",
                  f"| Average latency | {summary['avg_latency_s']} s |"]
        if summary["judge_faithfulness_avg"] is not None:
            lines += [f"| LLM judge: faithfulness (1-5) | {summary['judge_faithfulness_avg']} |",
                      f"| LLM judge: relevance (1-5) | {summary['judge_relevance_avg']} |"]
    lines += ["", "## Results per question", "", "| ID | Type | Question | Result | Pages retrieved | Note |",
              "|---|---|---|---|---|---|"]
    for r in rows:
        if r["skipped"]:
            result, note = "skipped", r.get("note", "")
        elif retrieval_only:
            result = {True: "hit", False: "miss"}.get(r.get("passed")) or ("no match" if r.get("retrieval_refused") else "matched")
            note = f"best distance {r.get('best_distance')}"
        else:
            result = "PASS" if r["passed"] else "FAIL"
            note = f"status {r['status']}, citations {r['citations']}, tries {r['attempts']}"
        lines.append(f"| {r['id']} | {r['type']} | {_cell(r['question'], 70)} | {result} | "
                     f"{r.get('retrieved_pages', '')} | {_cell(note, 80)} |")
    if not retrieval_only:
        failed = [r for r in rows if not r["skipped"] and not r["passed"]]
        lines += ["", "## Failed cases (bad outputs)", ""]
        if not failed:
            lines.append("None. Every test case passed.")
        for r in failed:
            lines += [f"- **{r['id']}** ({r['type']}): {r['question']}",
                      f"  - answer: {_cell(r['answer'], 300)}",
                      f"  - found={r['found']}, status={r['status']}, citations={r['citations']}, "
                      f"retrieved pages={r['retrieved_pages']}"]
        lines += ["", "## Example answers (good outputs)", ""]
        for r in [r for r in rows if not r["skipped"] and r["passed"] and r["type"] == "answerable"][:4]:
            lines += [f"- **{r['id']}** {r['question']}", f"  - {_cell(r['answer'], 300)} (pages {r['citations']})"]
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate the PDF assistant with a golden test set.")
    parser.add_argument("--pdf", default=str(ROOT / "data" / "matrix_basics.pdf"))
    parser.add_argument("--golden", default=str(DEFAULT_GOLDEN))
    parser.add_argument("--out", default=str(RESULTS_DIR))
    parser.add_argument("--retrieval-only", action="store_true", help="skip the LLM (no API key needed)")
    parser.add_argument("--judge", action="store_true", help="add LLM-as-judge scores")
    parser.add_argument("--delay", type=float, default=1.0, help="seconds to wait between questions (rate limits)")
    args = parser.parse_args(argv)
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    from rag.factory import build_assistant
    from rag.llm import explain_error
    try:
        assistant = build_assistant(need_llm=not args.retrieval_only)
        print("Indexed:", assistant.index_pdf(args.pdf))
    except Exception as error:
        print(f"Could not start: {explain_error(error)}", file=sys.stderr)
        return 1

    golden = json.loads(Path(args.golden).read_text(encoding="utf-8"))
    rows, summary = run(assistant, golden, args.retrieval_only, args.judge, args.delay)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    suffix = "_retrieval_only" if args.retrieval_only else ""
    (out / f"report{suffix}.md").write_text(build_report(rows, summary, assistant.config, args.retrieval_only), encoding="utf-8")
    (out / f"latest{suffix}.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=2), encoding="utf-8")
    print("\nSummary:", json.dumps(summary, indent=2))
    print(f"Report written to {out / ('report' + suffix + '.md')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
