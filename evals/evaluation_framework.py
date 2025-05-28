# evaluation_framework.py - Professional Prompt Evaluation Framework
"""
Evaluation framework for prompt lab experiments.
Provides standardized metrics and comparison tools.
"""

import json
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path

class PromptEvaluator:
    """Professional evaluation framework for prompt experiments."""
    
    def __init__(self, eval_name: str, output_dir: str = "../data/evaluations"):
        self.eval_name = eval_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results = []
        
    def evaluate_response(
        self,
        prompt: str,
        response: str,
        expected: Optional[str] = None,
        criteria: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a single prompt-response pair.
        
        Args:
            prompt: The input prompt
            response: The model's response
            expected: Expected response (if available)
            criteria: Evaluation criteria dictionary
            metadata: Additional metadata (model, tokens, cost, etc.)
        
        Returns:
            Evaluation result dictionary
        """
        result = {
            'timestamp': datetime.now().isoformat(),
            'prompt': prompt,
            'response': response,
            'expected': expected,
            'metadata': metadata or {},
            'scores': {}
        }
        
        # Basic metrics
        result['scores']['response_length'] = len(response.split())
        result['scores']['response_chars'] = len(response)
        
        # Quality metrics (if criteria provided)
        if criteria:
            for criterion, target in criteria.items():
                if criterion == 'min_length':
                    result['scores'][f'{criterion}_pass'] = len(response.split()) >= target
                elif criterion == 'max_length':
                    result['scores'][f'{criterion}_pass'] = len(response.split()) <= target
                elif criterion == 'contains_keywords':
                    keywords_found = sum(1 for kw in target if kw.lower() in response.lower())
                    result['scores']['keywords_found'] = keywords_found
                    result['scores']['keywords_total'] = len(target)
                    result['scores']['keyword_coverage'] = keywords_found / len(target) if target else 0
        
        # Exact match (if expected provided)
        if expected:
            result['scores']['exact_match'] = response.strip() == expected.strip()
            result['scores']['length_similarity'] = min(len(response), len(expected)) / max(len(response), len(expected))
        
        self.results.append(result)
        return result
    
    def batch_evaluate(
        self,
        test_cases: List[Dict[str, Any]],
        model_runner,
        model_name: str,
        phase: str = "evaluation"
    ) -> pd.DataFrame:
        """
        Evaluate multiple test cases in batch.
        
        Args:
            test_cases: List of test case dictionaries
            model_runner: Function to call model
            model_name: Name of the model being tested
            phase: Evaluation phase name
        
        Returns:
            DataFrame with results
        """
        print(f"🔍 Running batch evaluation: {self.eval_name}")
        print(f"📊 Test cases: {len(test_cases)}")
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"⏳ Processing {i}/{len(test_cases)}: {test_case.get('name', f'Test {i}')}")
            
            try:
                # Run the model
                response, metrics = model_runner(
                    model=model_name,
                    messages=test_case['messages'],
                    phase=phase
                )
                
                # Evaluate the response
                self.evaluate_response(
                    prompt=str(test_case['messages']),
                    response=response,
                    expected=test_case.get('expected'),
                    criteria=test_case.get('criteria'),
                    metadata={
                        'test_name': test_case.get('name'),
                        'model': model_name,
                        'phase': phase,
                        **metrics
                    }
                )
                
            except Exception as e:
                print(f"❌ Error in test case {i}: {e}")
                continue
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame(self.results)
        return df
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive evaluation report."""
        if not self.results:
            return {"error": "No results to analyze"}
        
        df = pd.DataFrame(self.results)
        
        report = {
            'evaluation_name': self.eval_name,
            'timestamp': datetime.now().isoformat(),
            'total_tests': len(self.results),
            'summary_stats': {
                'avg_response_length': df['scores.response_length'].mean(),
                'avg_response_chars': df['scores.response_chars'].mean(),
            },
            'cost_analysis': {},
            'quality_metrics': {}
        }
        
        # Cost analysis if metadata available
        if any('cost_usd' in r.get('metadata', {}) for r in self.results):
            costs = [r['metadata'].get('cost_usd', 0) for r in self.results]
            report['cost_analysis'] = {
                'total_cost': sum(costs),
                'avg_cost_per_test': sum(costs) / len(costs),
                'min_cost': min(costs),
                'max_cost': max(costs)
            }
        
        # Quality metrics if available
        if 'scores.exact_match' in df.columns:
            report['quality_metrics']['exact_match_rate'] = df['scores.exact_match'].mean()
        
        if 'scores.keyword_coverage' in df.columns:
            report['quality_metrics']['avg_keyword_coverage'] = df['scores.keyword_coverage'].mean()
        
        return report
    
    def save_results(self) -> str:
        """Save results to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results
        results_file = self.output_dir / f"{self.eval_name}_{timestamp}_results.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Save summary report
        report = self.generate_report()
        report_file = self.output_dir / f"{self.eval_name}_{timestamp}_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Save CSV for analysis
        if self.results:
            df = pd.DataFrame(self.results)
            csv_file = self.output_dir / f"{self.eval_name}_{timestamp}_data.csv"
            df.to_csv(csv_file, index=False)
        
        print(f"💾 Results saved:")
        print(f"   📄 Detailed: {results_file}")
        print(f"   📊 Report: {report_file}")
        print(f"   📈 CSV: {csv_file}")
        
        return str(report_file)
