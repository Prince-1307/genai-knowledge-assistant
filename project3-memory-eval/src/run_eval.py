import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ask import ask, retrieve_chunks, generate, conversation_history
from eval_set import EVAL_SET


def judge_answer(question, expected, actual):
    """Uses the LLM to judge if actual answer matches expected, semantically."""
    prompt = f"""You are evaluating an AI system's answer for correctness.

Question: {question}
Expected answer: {expected}
Actual answer: {actual}

Does the actual answer correctly convey the expected answer? Minor formatting/phrasing differences are OK.
Respond with ONLY "CORRECT" or "INCORRECT", nothing else."""

    verdict = generate(prompt).strip()
    return verdict


def run_evaluation():
    results = []

    for item in EVAL_SET:
        conversation_history.clear()  # reset memory between eval questions, treat each independently
        actual = ask(item["question"])
        verdict = judge_answer(item["question"], item["expected"], actual)

        results.append({
            "question": item["question"],
            "expected": item["expected"],
            "actual": actual,
            "verdict": verdict
        })

    correct = sum(1 for r in results if "CORRECT" in r["verdict"].upper() and "INCORRECT" not in r["verdict"].upper())
    total = len(results)

    print(f"\n\n=== EVAL RESULTS: {correct}/{total} correct ({correct/total*100:.1f}%) ===\n")
    for r in results:
        status = "✅" if "CORRECT" in r["verdict"].upper() and "INCORRECT" not in r["verdict"].upper() else "❌"
        print(f"{status} Q: {r['question']}")
        print(f"   Expected: {r['expected']}")
        print(f"   Got: {r['actual']}")
        print()

    return results


if __name__ == "__main__":
    run_evaluation()