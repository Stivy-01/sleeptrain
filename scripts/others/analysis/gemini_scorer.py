"""
SleepTrain - Gemini-based Intelligent Scorer
Uses Gemini API to semantically evaluate responses instead of keyword matching.
"""

import json
import os
from pathlib import Path
import google.generativeai as genai
from typing import Dict, List, Optional
import time

# Load from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ Loaded .env file")
except ImportError:
    print("⚠️ python-dotenv not installed")

# Configure Gemini
def setup_gemini(api_key: str = None):
    """Setup Gemini API."""
    if api_key is None:
        api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found. Set it as environment variable or pass directly.")
    
    genai.configure(api_key=api_key)
    return genai.GenerativeModel('gemini-2.0-flash')


def score_with_gemini(
    model,
    question: str,
    response: str,
    expected_facts: List[str],
    person_name: str = ""
) -> Dict:
    """
    Use Gemini to intelligently score a response.
    
    Returns:
        {
            "score": 0.0-1.0,
            "correct_facts": [...],
            "missing_facts": [...],
            "incorrect_facts": [...],
            "reasoning": "..."
        }
    """
    
    expected_str = "\n".join([f"- {fact}" for fact in expected_facts])
    
    prompt = f"""You are evaluating an AI's response for factual accuracy.

QUESTION: {question}

AI RESPONSE: {response}

EXPECTED FACTS (the response should contain these):
{expected_str}

TASK: Evaluate if the response correctly conveys the expected facts.
Be LENIENT on format - "2009" and "January 2009" both count as correct.
Be STRICT on factual accuracy - wrong dates/names are incorrect.

Return ONLY valid JSON:
{{
    "score": <0.0 to 1.0 based on how many expected facts are correctly conveyed>,
    "correct_facts": ["list of facts from expected that ARE correctly stated"],
    "missing_facts": ["list of facts from expected that are NOT mentioned"],
    "incorrect_facts": ["list of factually WRONG statements in the response"],
    "reasoning": "brief explanation"
}}"""

    try:
        result = model.generate_content(prompt)
        text = result.text.strip()
        
        # Extract JSON
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()
        
        # Find JSON bounds
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            text = text[start:end]
        
        return json.loads(text)
    
    except Exception as e:
        print(f"⚠️ Gemini scoring error: {e}")
        return {
            "score": 0.0,
            "correct_facts": [],
            "missing_facts": expected_facts,
            "incorrect_facts": [],
            "reasoning": f"Error: {str(e)}"
        }


def score_correction_with_gemini(
    model,
    question: str,
    response: str,
    wrong_info: str,
    correct_info: str,
    person_name: str = ""
) -> Dict:
    """
    Use Gemini to score correction questions.
    
    Checks:
    1. Did the model indicate the info was WRONG?
    2. Did the model provide the CORRECT info?
    """
    
    prompt = f"""You are evaluating if an AI correctly rejected false information.

QUESTION (contains WRONG info): {question}

WRONG INFO in question: {wrong_info}
CORRECT INFO should be: {correct_info}

AI RESPONSE: {response}

TASK: Evaluate:
1. Did the AI clearly indicate the information was WRONG/INCORRECT? (said "no", "incorrect", "wrong", etc.)
2. Did the AI provide the CORRECT information ({correct_info})?

Be LENIENT on date formats - "2009" and "January 2009" both count.
Be STRICT on whether they actually corrected the misinformation.

Return ONLY valid JSON:
{{
    "indicated_wrong": true/false,
    "provided_correct": true/false,
    "score": <0.0, 0.3, 0.7, or 1.0>,
    "reasoning": "brief explanation"
}}

Scoring:
- 1.0 = Said it's wrong AND gave correct info
- 0.7 = Gave correct info but didn't explicitly say "wrong/incorrect"
- 0.3 = Said it's wrong but didn't give correct info
- 0.0 = Neither"""

    try:
        result = model.generate_content(prompt)
        text = result.text.strip()
        
        # Extract JSON
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()
        
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            text = text[start:end]
        
        return json.loads(text)
    
    except Exception as e:
        print(f"⚠️ Gemini scoring error: {e}")
        return {
            "indicated_wrong": False,
            "provided_correct": False,
            "score": 0.0,
            "reasoning": f"Error: {str(e)}"
        }


def rescore_experiment(experiment_path: str, api_key: str = None) -> Dict:
    """
    Re-score an experiment JSON file using Gemini.
    """
    
    print("🧠 Setting up Gemini scorer...")
    model = setup_gemini(api_key)
    
    print(f"📂 Loading experiment: {experiment_path}")
    with open(experiment_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    results = {
        "original_file": experiment_path,
        "rescored_with": "gemini-2.0-flash",
        "conversation_6turn": {"turns": [], "original_score": 0, "new_score": 0},
        "correction_test": {"questions": [], "original_score": 0, "new_score": 0},
        "extended_test": {"sample_turns": [], "original_score": 0, "new_score": 0},
    }
    
    # Re-score 6-turn conversation
    print("\n📊 Re-scoring 6-turn conversation...")
    convo = data["tests"].get("conversation_6turn", {})
    turns = convo.get("turns", [])
    new_convo_scores = []
    
    for i, turn in enumerate(turns):
        print(f"   Turn {i+1}/{len(turns)}: {turn['question'][:40]}...")
        
        score_result = score_with_gemini(
            model,
            question=turn["question"],
            response=turn["response"],
            expected_facts=turn.get("expected_keywords", []),
            person_name=turn.get("person", "")
        )
        
        new_convo_scores.append(score_result["score"])
        results["conversation_6turn"]["turns"].append({
            "turn": turn["turn"],
            "question": turn["question"],
            "original_score": turn["score"],
            "new_score": score_result["score"],
            "reasoning": score_result["reasoning"]
        })
        
        time.sleep(0.5)  # Rate limiting
    
    results["conversation_6turn"]["original_score"] = convo.get("overall_score", 0)
    results["conversation_6turn"]["new_score"] = sum(new_convo_scores) / len(new_convo_scores) if new_convo_scores else 0
    
    # Re-score correction test
    print("\n🔧 Re-scoring correction test...")
    correction = data["tests"].get("correction_test", {})
    questions = correction.get("questions", [])
    new_correction_scores = []
    
    for i, q in enumerate(questions):
        print(f"   Q{i+1}/{len(questions)}: {q['question'][:40]}...")
        
        score_result = score_correction_with_gemini(
            model,
            question=q["question"],
            response=q["response"],
            wrong_info=q.get("wrong_date", ""),
            correct_info=q.get("correct_date", ""),
            person_name=q.get("person", "")
        )
        
        new_correction_scores.append(score_result["score"])
        results["correction_test"]["questions"].append({
            "question": q["question"],
            "original_score": q["score"],
            "new_score": score_result["score"],
            "indicated_wrong": score_result["indicated_wrong"],
            "provided_correct": score_result["provided_correct"],
            "reasoning": score_result["reasoning"]
        })
        
        time.sleep(0.5)
    
    results["correction_test"]["original_score"] = correction.get("avg_score", 0)
    results["correction_test"]["new_score"] = sum(new_correction_scores) / len(new_correction_scores) if new_correction_scores else 0
    
    # Sample re-score of extended test (first 10 turns to save API calls)
    print("\n🔄 Re-scoring extended test (sample of 10 turns)...")
    extended = data["tests"].get("extended_test", {})
    ext_turns = extended.get("turns", [])[:10]
    new_ext_scores = []
    
    for i, turn in enumerate(ext_turns):
        print(f"   Turn {i+1}/{len(ext_turns)}: {turn['question'][:40]}...")
        
        if turn.get("type") == "correction":
            # This is a correction question
            score_result = score_correction_with_gemini(
                model,
                question=turn["question"],
                response=turn["response"],
                wrong_info="wrong date in question",
                correct_info=", ".join(turn.get("expected", [])),
                person_name=turn.get("person", "")
            )
        else:
            score_result = score_with_gemini(
                model,
                question=turn["question"],
                response=turn["response"],
                expected_facts=turn.get("expected", []),
                person_name=turn.get("person", "")
            )
        
        new_ext_scores.append(score_result["score"])
        results["extended_test"]["sample_turns"].append({
            "turn": turn["turn"],
            "question": turn["question"],
            "original_score": turn["score"],
            "new_score": score_result["score"],
            "reasoning": score_result.get("reasoning", "")
        })
        
        time.sleep(0.5)
    
    results["extended_test"]["original_score"] = extended.get("overall_avg", 0)
    results["extended_test"]["new_score"] = sum(new_ext_scores) / len(new_ext_scores) if new_ext_scores else 0
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 GEMINI RE-SCORING SUMMARY")
    print("=" * 60)
    
    print(f"\n6-Turn Conversation:")
    print(f"   Original: {results['conversation_6turn']['original_score']:.1%}")
    print(f"   Gemini:   {results['conversation_6turn']['new_score']:.1%}")
    diff = results['conversation_6turn']['new_score'] - results['conversation_6turn']['original_score']
    print(f"   Change:   {diff:+.1%}")
    
    print(f"\nCorrection Test:")
    print(f"   Original: {results['correction_test']['original_score']:.1%}")
    print(f"   Gemini:   {results['correction_test']['new_score']:.1%}")
    diff = results['correction_test']['new_score'] - results['correction_test']['original_score']
    print(f"   Change:   {diff:+.1%}")
    
    print(f"\nExtended Test (10-turn sample):")
    print(f"   Original: {results['extended_test']['original_score']:.1%}")
    print(f"   Gemini:   {results['extended_test']['new_score']:.1%}")
    diff = results['extended_test']['new_score'] - results['extended_test']['original_score']
    print(f"   Change:   {diff:+.1%}")
    
    # Save results to the organized rescored directory
    experiment_file = Path(experiment_path)
    script_dir = Path(__file__).parent.parent.parent.parent  # Go up to project root
    output_dir = script_dir / "data" / "experiment_results" / "training" / "rescored"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_filename = experiment_file.stem + "_gemini_rescored.json"
    output_path = output_dir / output_filename
    
    with open(str(output_path), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_path}")
    
    return results


# For use in notebook
def create_gemini_scorer_for_notebook():
    """
    Returns functions that can be used directly in the notebook.
    """
    return """
# Gemini Scorer for Notebook
# Add this to your notebook to use Gemini for scoring

import google.generativeai as genai
import json

def setup_scorer():
    from google.colab import userdata
    api_key = userdata.get('GEMINI_API_KEY')
    genai.configure(api_key=api_key)
    return genai.GenerativeModel('gemini-2.0-flash')

SCORER_MODEL = setup_scorer()

def gemini_score(question, response, expected_keywords, is_correction=False, wrong_info="", correct_info=""):
    '''Smart scoring using Gemini'''
    
    if is_correction:
        prompt = f'''Evaluate if this response correctly rejects false information.
Question (has WRONG info): {question}
Wrong info: {wrong_info}
Correct info: {correct_info}
Response: {response}

Did it: 1) Say the info was wrong? 2) Give the correct info?
Return JSON: {{"score": 0.0-1.0, "reasoning": "brief"}}
1.0 = both, 0.7 = correct info only, 0.3 = said wrong only, 0.0 = neither'''
    else:
        expected = ", ".join(expected_keywords)
        prompt = f'''Evaluate factual accuracy.
Question: {question}
Expected facts: {expected}
Response: {response}

Be LENIENT on format ("2009" = "January 2009").
Return JSON: {{"score": 0.0-1.0, "reasoning": "brief"}}'''
    
    try:
        result = SCORER_MODEL.generate_content(prompt)
        text = result.text
        start, end = text.find("{"), text.rfind("}") + 1
        return json.loads(text[start:end])["score"]
    except:
        return 0.0  # Fallback
"""


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python gemini_scorer.py <experiment.json> [GEMINI_API_KEY]")
        print("\nOr set GEMINI_API_KEY environment variable")
        sys.exit(1)
    
    experiment_path = sys.argv[1]
    api_key = sys.argv[2] if len(sys.argv) > 2 else None
    
    rescore_experiment(experiment_path, api_key)
