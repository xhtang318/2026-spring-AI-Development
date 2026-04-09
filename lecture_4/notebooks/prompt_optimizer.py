"""
Prompt optimization pipeline for resume scoring.

Iteratively improves a scoring prompt to achieve target score ranges:
  - Gold resumes (g01, g02): 85+
  - Silver resumes (s01, s02): 70-85
  - Wild resumes: much lower (below 50 ideally)

Uses a meta-prompt (with a strong model) to analyze failures and rewrite
the scoring prompt, then tests the new prompt with the target model.
"""

import json
import sys
import random
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

OPENROUTER_API_KEY = ""  # Set via env or paste here
SCORING_MODEL = "mistralai/mistral-small-2603"
OPTIMIZER_MODEL = "anthropic/claude-sonnet-4-6"
MAX_ITERATIONS = 8
TEMPERATURE = 0.3

# Target ranges
GOLD_MIN = 85
SILVER_MIN = 70
SILVER_MAX = 85
WILD_MAX = 50

# ---------------------------------------------------------------------------
# Data loading (reuse from lecture 3)
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_data():
    """Load resumes and job requirements."""
    import csv

    csv_path = DATA_DIR / "resumes_final_with_gold_silver.csv"
    resumes = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            resumes[row["ID"]] = {
                "ID": row["ID"],
                "Resume_str": row["Resume_str"],
            }

    job_req_path = DATA_DIR / "job_req_senior.md"
    with open(job_req_path, "r", encoding="utf-8") as f:
        job_req = f.read()

    # Select test set: all gold + silver + 2 wild
    gold = {rid: r for rid, r in resumes.items() if rid.startswith("g")}
    silver = {rid: r for rid, r in resumes.items() if rid.startswith("s")}
    wild = {rid: r for rid, r in resumes.items() if rid[0].isdigit()}
    random.seed(42)
    wild_sample = dict(random.sample(list(wild.items()), 2))

    return gold, silver, wild_sample, job_req


# ---------------------------------------------------------------------------
# LLM call helpers
# ---------------------------------------------------------------------------

class ResumeScore(BaseModel):
    score: int = Field(..., ge=0, le=100)
    reasoning: str = Field(...)


def _clean_schema(obj):
    if isinstance(obj, dict):
        return {
            k: _clean_schema(v)
            for k, v in obj.items()
            if k not in ("title", "minimum", "maximum", "exclusiveMinimum",
                         "exclusiveMaximum", "default")
        }
    if isinstance(obj, list):
        return [_clean_schema(item) for item in obj]
    return obj


def score_resume(api_key: str, prompt: str, resume_text: str, model: str) -> dict:
    """Score a single resume using the given prompt."""
    schema = _clean_schema(ResumeScore.model_json_schema())
    schema["additionalProperties"] = False

    full_prompt = f"""{prompt}

Resume:
{resume_text[:3000]}"""

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": full_prompt}],
        "temperature": TEMPERATURE,
        "max_tokens": 1500,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "ResumeScore",
                "strict": True,
                "schema": schema,
            },
        },
    }

    import time
    for attempt in range(3):
        try:
            with httpx.Client(timeout=60) as client:
                resp = client.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                )
                resp.raise_for_status()
                data = resp.json()

                if "error" in data:
                    return {"score": None, "reasoning": None, "error": str(data["error"])}

                content = data["choices"][0]["message"]["content"]
                parsed = ResumeScore.model_validate_json(content)
                cost = data.get("usage", {}).get("cost")
                return {"score": parsed.score, "reasoning": parsed.reasoning, "error": None, "cost": cost}
        except Exception as e:
            if attempt < 2 and ("503" in str(e) or "502" in str(e) or "timeout" in str(e).lower()):
                time.sleep(2 * (attempt + 1))
                continue
            return {"score": None, "reasoning": None, "error": str(e)}


def run_evaluation(api_key: str, prompt: str, gold: dict, silver: dict, wild: dict, job_req: str, model: str) -> dict:
    """Run the prompt on all test resumes and return results."""
    results = {}
    total_cost = 0.0

    for category, resumes in [("gold", gold), ("silver", silver), ("wild", wild)]:
        for rid, resume in sorted(resumes.items()):
            resp = score_resume(api_key, prompt, resume["Resume_str"], model)
            results[rid] = {
                "category": category,
                "score": resp["score"],
                "reasoning": resp.get("reasoning", ""),
                "error": resp.get("error"),
            }
            if resp.get("cost"):
                total_cost += resp["cost"]
            status = f"{resp['score']}" if resp["score"] is not None else f"ERROR: {resp['error']}"
            print(f"    {rid} ({category}): {status}")

    results["_total_cost"] = total_cost
    return results


def evaluate_results(results: dict) -> dict:
    """Check if results meet targets. Returns analysis dict."""
    gold_scores = [r["score"] for rid, r in results.items() if not rid.startswith("_") and r["category"] == "gold" and r["score"] is not None]
    silver_scores = [r["score"] for rid, r in results.items() if not rid.startswith("_") and r["category"] == "silver" and r["score"] is not None]
    wild_scores = [r["score"] for rid, r in results.items() if not rid.startswith("_") and r["category"] == "wild" and r["score"] is not None]

    gold_mean = sum(gold_scores) / len(gold_scores) if gold_scores else 0
    silver_mean = sum(silver_scores) / len(silver_scores) if silver_scores else 0
    wild_mean = sum(wild_scores) / len(wild_scores) if wild_scores else 0

    gold_ok = all(s >= GOLD_MIN for s in gold_scores) if gold_scores else False
    silver_ok = all(SILVER_MIN <= s <= SILVER_MAX for s in silver_scores) if silver_scores else False
    wild_ok = all(s <= WILD_MAX for s in wild_scores) if wild_scores else False

    problems = []
    for rid, r in sorted(results.items()):
        if rid.startswith("_"):
            continue
        s = r["score"]
        if s is None:
            problems.append(f"{rid}: ERROR - {r['error']}")
            continue
        if r["category"] == "gold" and s < GOLD_MIN:
            problems.append(f"{rid} (gold): scored {s}, need {GOLD_MIN}+. Reasoning: {r['reasoning'][:200]}")
        if r["category"] == "silver" and (s < SILVER_MIN or s > SILVER_MAX):
            problems.append(f"{rid} (silver): scored {s}, need {SILVER_MIN}-{SILVER_MAX}. Reasoning: {r['reasoning'][:200]}")
        if r["category"] == "wild" and s > WILD_MAX:
            problems.append(f"{rid} (wild): scored {s}, need <={WILD_MAX}. Reasoning: {r['reasoning'][:200]}")

    passed = gold_ok and silver_ok and wild_ok

    return {
        "passed": passed,
        "gold_mean": gold_mean,
        "silver_mean": silver_mean,
        "wild_mean": wild_mean,
        "gold_scores": gold_scores,
        "silver_scores": silver_scores,
        "wild_scores": wild_scores,
        "problems": problems,
    }


# ---------------------------------------------------------------------------
# Meta-prompt: ask a strong model to improve the scoring prompt
# ---------------------------------------------------------------------------

def optimize_prompt(api_key: str, current_prompt: str, evaluation: dict, job_req: str, iteration: int) -> str:
    """Use a strong model to rewrite the scoring prompt based on failures."""

    meta_prompt = f"""You are an expert prompt engineer. Your task is to improve a resume scoring prompt.

The scoring prompt is used with a SMALL language model ({SCORING_MODEL}) to score resumes against job requirements on a 0-100 scale. The model returns JSON with "score" and "reasoning" fields.

CURRENT SCORING PROMPT:
---
{current_prompt}
---

JOB REQUIREMENTS SUMMARY:
Full-Stack Senior Software Engineer (.NET/C#, JS, SQL, AWS). Requires 5+ years C#/.NET, frontend JS framework, strong SQL, Git/CI-CD. Preferred: AWS, microservices, security knowledge.

TARGET SCORES:
- Gold resumes (strong full-stack .NET developers with 8-10 years, AWS, SQL Server, React/Angular): {GOLD_MIN}+
- Silver resumes (one is a 1.5yr junior dev, other is a 10yr frontend-only specialist with no .NET/SQL): {SILVER_MIN}-{SILVER_MAX}
- Wild resumes (random, mostly poor matches): below {WILD_MAX}

CURRENT RESULTS:
- Gold scores: {evaluation['gold_scores']} (mean: {evaluation['gold_mean']:.1f}, target: {GOLD_MIN}+)
- Silver scores: {evaluation['silver_scores']} (mean: {evaluation['silver_mean']:.1f}, target: {SILVER_MIN}-{SILVER_MAX})
- Wild scores: {evaluation['wild_scores']} (mean: {evaluation['wild_mean']:.1f}, target: <{WILD_MAX})

PROBLEMS TO FIX:
{chr(10).join(f'- {p}' for p in evaluation['problems']) if evaluation['problems'] else '- None (all targets met!)'}

This is iteration {iteration}/{MAX_ITERATIONS}. Write an improved scoring prompt that fixes these problems.

IMPORTANT RULES:
1. The prompt must include the job requirements inline (use a placeholder {{job_req}} that will be filled in)
2. The prompt must ask for a score 0-100 and brief reasoning
3. Be very specific about what makes a score high vs low vs medium
4. Include scoring criteria with clear anchor points
5. For small models, be explicit and direct - don't rely on subtle hints
6. The prompt should penalize heavily for missing CORE requirements (.NET/C#, SQL, 5+ years)
7. Partial matches (e.g., frontend-only with no backend) should score 70-85, not 85+
8. Junior developers with <3 years should score in the 70-85 range (they show potential but lack experience)

Return ONLY the improved prompt text. No explanation or wrapper."""

    try:
        with httpx.Client(timeout=120) as client:
            resp = client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": OPTIMIZER_MODEL,
                    "messages": [{"role": "user", "content": meta_prompt}],
                    "temperature": 0.7,
                    "max_tokens": 3000,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"  ERROR optimizing prompt: {e}")
        return current_prompt


# ---------------------------------------------------------------------------
# Initial prompt
# ---------------------------------------------------------------------------

INITIAL_PROMPT = """Score this resume against the job requirements on a 0-100 scale.
Provide a score and brief reasoning.

Job Requirements: {job_req}"""


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    import os
    api_key = os.environ.get("OPENROUTER_API_KEY", OPENROUTER_API_KEY)
    if not api_key:
        print("ERROR: Set OPENROUTER_API_KEY environment variable or edit the script.")
        sys.exit(1)

    print("Loading data...")
    gold, silver, wild, job_req = load_data()
    print(f"Test set: {len(gold)} gold, {len(silver)} silver, {len(wild)} wild\n")

    current_prompt = INITIAL_PROMPT.replace("{job_req}", job_req)
    best_prompt = current_prompt
    best_score = -999  # composite quality metric

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"{'='*60}")
        print(f"ITERATION {iteration}/{MAX_ITERATIONS}")
        print(f"{'='*60}")
        print(f"\nPrompt ({len(current_prompt)} chars):")
        print(f"  {current_prompt[:200]}...")
        print(f"\nScoring with {SCORING_MODEL}:")

        # Run evaluation
        results = run_evaluation(api_key, current_prompt, gold, silver, wild, job_req, SCORING_MODEL)
        evaluation = evaluate_results(results)
        total_cost = results.get("_total_cost", 0)

        # Print summary
        print(f"\n  Results:")
        print(f"    Gold:   {evaluation['gold_scores']} (mean: {evaluation['gold_mean']:.1f}, target: {GOLD_MIN}+)")
        print(f"    Silver: {evaluation['silver_scores']} (mean: {evaluation['silver_mean']:.1f}, target: {SILVER_MIN}-{SILVER_MAX})")
        print(f"    Wild:   {evaluation['wild_scores']} (mean: {evaluation['wild_mean']:.1f}, target: <{WILD_MAX})")
        print(f"    Cost:   ${total_cost:.5f}")

        if evaluation["problems"]:
            print(f"\n  Problems:")
            for p in evaluation["problems"]:
                print(f"    - {p[:120]}")

        # Composite quality score: gold_mean - penalties
        quality = evaluation["gold_mean"]
        for s in evaluation["silver_scores"]:
            if s < SILVER_MIN:
                quality -= (SILVER_MIN - s)
            elif s > SILVER_MAX:
                quality -= (s - SILVER_MAX)
        for s in evaluation["wild_scores"]:
            if s > WILD_MAX:
                quality -= (s - WILD_MAX)
        for s in evaluation["gold_scores"]:
            if s < GOLD_MIN:
                quality -= (GOLD_MIN - s) * 2  # heavier penalty for gold misses

        if quality > best_score:
            best_score = quality
            best_prompt = current_prompt
            print(f"\n  *** New best prompt (quality: {quality:.1f}) ***")

        if evaluation["passed"]:
            print(f"\n{'='*60}")
            print(f"SUCCESS! All targets met on iteration {iteration}.")
            print(f"{'='*60}")
            break

        if iteration < MAX_ITERATIONS:
            print(f"\n  Optimizing prompt with {OPTIMIZER_MODEL}...")
            new_prompt = optimize_prompt(api_key, current_prompt, evaluation, job_req, iteration)
            # Ensure job_req is included if the optimizer used the placeholder
            if "{job_req}" in new_prompt:
                new_prompt = new_prompt.replace("{job_req}", job_req)
            current_prompt = new_prompt

    # Final output
    print(f"\n{'='*60}")
    print(f"BEST PROMPT (quality score: {best_score:.1f}):")
    print(f"{'='*60}")
    print(best_prompt)

    # Save to file
    output_path = Path(__file__).resolve().parent / "optimized_prompt.txt"
    with open(output_path, "w") as f:
        f.write(best_prompt)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
