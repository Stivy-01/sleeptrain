# scripts/analysis/human_eval.py

"""
Human evaluation interface.

Collect human ratings of model responses for validation.
"""

import json
from typing import List, Dict
import random

class HumanEvaluator:
    """Interface for collecting human ratings."""
    
    def __init__(self, output_path: str = "human_eval_results.jsonl"):
        """
        Initialize evaluator.
        
        Args:
            output_path: Where to save ratings
        """
        self.output_path = output_path
        self.results = []
    
    def evaluate_responses(
        self,
        people: List[Dict],
        model_responses: Dict[str, str],
        gemini_scores: Dict[str, Dict]
    ) -> Dict[str, any]:
        """
        Collect human ratings for model responses.
        
        Args:
            people: List of person dicts
            model_responses: Dict of person_id -> response text
            gemini_scores: Dict of person_id -> automated scores
            
        Returns:
            Dict with inter-rater reliability and human vs auto comparison
        """
        print("\n" + "="*70)
        print("🧑 HUMAN EVALUATION")
        print("="*70)
        print("\nRate each response on a scale of 1-10:")
        print("  1-3: Poor (missing most facts or wrong)")
        print("  4-6: Fair (some facts correct)")
        print("  7-8: Good (most facts correct)")
        print("  9-10: Excellent (all facts correct and well-expressed)")
        print()
        
        for person in people:
            pid = person["id"]
            name = person["name"]
            response = model_responses.get(pid, "")
            auto_score = gemini_scores.get(pid, {}).get("overall", 0)
            
            print(f"\n{'='*70}")
            print(f"PERSON: {name}")
            print(f"{'='*70}")
            
            # Show facts to check
            print(f"\n📋 Expected facts:")
            for i, fact in enumerate(person["facts"][:5], 1):
                print(f"   {i}. {fact['fact']}")
            
            # Show response
            print(f"\n🤖 Model response:")
            print(f"   {response}")
            
            # Show automated score
            print(f"\n🤖 Automated score: {auto_score:.1%}")
            
            # Get human rating
            while True:
                try:
                    rating = int(input(f"\n👤 Your rating (1-10): "))
                    if 1 <= rating <= 10:
                        break
                    print("   Please enter a number between 1 and 10")
                except ValueError:
                    print("   Please enter a valid number")
            
            # Collect specific feedback
            completeness = input(f"   Completeness (1-5): ")
            accuracy = input(f"   Accuracy (1-5): ")
            fluency = input(f"   Fluency (1-5): ")
            
            # Save result
            result = {
                "person": pid,
                "name": name,
                "response": response,
                "human_rating": rating,
                "human_completeness": int(completeness) if completeness.isdigit() else None,
                "human_accuracy": int(accuracy) if accuracy.isdigit() else None,
                "human_fluency": int(fluency) if fluency.isdigit() else None,
                "automated_score": auto_score
            }
            
            self.results.append(result)
        
        # Save results
        self._save_results()
        
        # Calculate statistics
        stats = self._calculate_stats()
        
        return stats
    
    def _save_results(self):
        """Save results to file."""
        with open(self.output_path, 'w') as f:
            for result in self.results:
                f.write(json.dumps(result) + '\n')
        print(f"\n✅ Results saved to {self.output_path}")
    
    def _calculate_stats(self) -> Dict:
        """Calculate inter-rater reliability and correlation."""
        import numpy as np
        from scipy.stats import spearmanr, pearsonr
        
        human_ratings = [r["human_rating"] / 10.0 for r in self.results]  # Normalize to 0-1
        auto_scores = [r["automated_score"] for r in self.results]
        
        # Correlation
        pearson_r, pearson_p = pearsonr(human_ratings, auto_scores)
        spearman_r, spearman_p = spearmanr(human_ratings, auto_scores)
        
        # Mean absolute error
        mae = np.mean([abs(h - a) for h, a in zip(human_ratings, auto_scores)])
        
        stats = {
            "n_samples": len(self.results),
            "human_mean": np.mean(human_ratings),
            "human_std": np.std(human_ratings),
            "auto_mean": np.mean(auto_scores),
            "auto_std": np.std(auto_scores),
            "pearson_r": pearson_r,
            "pearson_p": pearson_p,
            "spearman_r": spearman_r,
            "spearman_p": spearman_p,
            "mae": mae
        }
        
        # Print report
        print("\n" + "="*70)
        print("📊 HUMAN EVALUATION RESULTS")
        print("="*70)
        print(f"\nSamples evaluated: {stats['n_samples']}")
        print(f"\nHuman ratings:     {stats['human_mean']:.1%} ± {stats['human_std']:.1%}")
        print(f"Automated scores:  {stats['auto_mean']:.1%} ± {stats['auto_std']:.1%}")
        print(f"\nCorrelation:")
        print(f"  Pearson r:  {stats['pearson_r']:.3f} (p={stats['pearson_p']:.4f})")
        print(f"  Spearman ρ: {stats['spearman_r']:.3f} (p={stats['spearman_p']:.4f})")
        print(f"\nMean Absolute Error: {stats['mae']:.1%}")
        
        # Interpretation
        if stats['pearson_r'] > 0.7:
            print("\n✅ Strong correlation - automated scoring is reliable")
        elif stats['pearson_r'] > 0.5:
            print("\n⚠️ Moderate correlation - automated scoring has some validity")
        else:
            print("\n❌ Weak correlation - automated scoring may not be reliable")
        
        return stats


# Example usage
if __name__ == "__main__":
    evaluator = HumanEvaluator()
    
    # Load results from experiment
    with open("final_results.json") as f:
        experiment = json.load(f)
    
    people = experiment["people"]
    model_responses = {p["id"]: experiment["final_recalls"][p["id"]] for p in people}
    gemini_scores = {p["id"]: experiment["final_scores"][p["id"]] for p in people}
    
    # Run human evaluation
    stats = evaluator.evaluate_responses(people, model_responses, gemini_scores)
