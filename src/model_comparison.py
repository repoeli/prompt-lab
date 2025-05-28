"""
Multi-Model Comparison Framework
================================

This module allows systematic comparison across different AI models.
Currently optimized for OpenAI models but designed to be extensible.

Key Features:
- Compare responses across multiple models
- Consistent evaluation criteria
- Cost-aware comparisons
- Extensible architecture for future model additions
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import os

def _get_openai():
    """Lazy import for openai to avoid potential import issues"""
    try:
        import openai
        return openai
    except ImportError:
        print("Warning: openai not available. Model comparison will be limited.")
        return None

class ModelComparisonFramework:
    """
    Framework for comparing responses across different AI models.
    
    Features:
    - Unified interface for different model providers
    - Cost tracking across models
        - Performance comparison metrics
    - Extensible design for future models
    """
    
    def __init__(self, comparison_name: str, output_dir: str = None):
        self.comparison_name = comparison_name
        if output_dir is None:
            # Get the project root directory (where this script is run from)
            project_root = Path(__file__).parent.parent
            output_dir = project_root / "data" / "model_comparisons"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = []
        self.model_configs = {}
          # Model pricing ($ per token) - easily extensible
        self.pricing = {
            "gpt-4o-mini": {"in": 0.60e-6, "out": 2.40e-6},
            "gpt-4o": {"in": 5.00e-6, "out": 15.00e-6},  # For future use
            "gpt-4": {"in": 30.00e-6, "out": 60.00e-6},   # For future use
            "gpt-3.5-turbo": {"in": 0.50e-6, "out": 1.50e-6},  # For future use
        }
        
        # Available models - can be extended
        self.available_models = {
            "gpt-4o-mini": self._call_openai_model,
            "gpt-4o": self._call_openai_model,
            "gpt-4": self._call_openai_model,
            "gpt-3.5-turbo": self._call_openai_model,
            # Future: Add other providers like Anthropic, Google, etc.
        }
    
    def add_model_config(self, model_name: str, config: Optional[Dict[str, Any]] = None):
        """
        Add a model to the comparison.
        
        Args:
            model_name: Name of the model (e.g., "gpt-4o-mini")
            config: Optional configuration (temperature, max_tokens, etc.)
        """
        if model_name not in self.available_models:
            print(f"⚠️  Model '{model_name}' not yet supported. Available: {list(self.available_models.keys())}")
            return False
        self.model_configs[model_name] = config or {}
        print(f"✅ Added model '{model_name}' to comparison")
        return True
        
    def _call_openai_model(self, model: str, messages: List[Dict], config: Dict) -> Tuple[str, Dict]:
        """
        Call OpenAI model with error handling and metrics collection.
        
        Args:
            model: Model name
            messages: List of message dictionaries
            config: Model configuration
            
        Returns:
            Tuple of (response_text, metrics_dict)
        """
        openai = _get_openai()
        if not openai:
            raise Exception("OpenAI not available")
            
        try:
            import time
            from src.runner import client  # Use existing client
            
            start_time = time.perf_counter()
            
            # Prepare API parameters
            api_params = {
                "model": model,
                "messages": messages,
                **config
            }
            
            response = client.chat.completions.create(**api_params)
            
            # Calculate metrics
            latency_ms = (time.perf_counter() - start_time) * 1000
            
            # Extract usage info
            usage = getattr(response, "usage", None)
            prompt_tokens = getattr(usage, "prompt_tokens", 0)
            completion_tokens = getattr(usage, "completion_tokens", 0)
            total_tokens = getattr(usage, "total_tokens", prompt_tokens + completion_tokens)
            
            # Calculate cost
            pricing = self.pricing.get(model, {"in": 0, "out": 0})
            cost = (prompt_tokens * pricing["in"]) + (completion_tokens * pricing["out"])
            
            response_text = response.choices[0].message.content
            
            metrics = {
                'prompt_tokens': prompt_tokens,
                'completion_tokens': completion_tokens,
                'total_tokens': total_tokens,
                'cost_usd': cost,
                'latency_ms': latency_ms,
                'model': model,
                'response_length': len(response_text.split()),
                'response_chars': len(response_text)            }
            
            return response_text, metrics
            
        except Exception as e:
            # Handle specific OpenAI errors if openai is available
            openai_mod = _get_openai()
            if openai_mod and hasattr(openai_mod, 'AuthenticationError') and isinstance(e, openai_mod.AuthenticationError):
                raise Exception(f"Authentication failed for {model}. Check your API key.")
            elif openai_mod and hasattr(openai_mod, 'RateLimitError') and isinstance(e, openai_mod.RateLimitError):
                raise Exception(f"Rate limit exceeded for {model}. Try again later.")
            else:
                raise Exception(f"Error calling {model}: {str(e)}")
    
    def run_comparison(
        self,
        test_prompts: List[Dict[str, Any]],
        iterations_per_model: int = 3
    ) -> pd.DataFrame:
        """
        Run comparison across all configured models.
        
        Args:
            test_prompts: List of prompt dictionaries with 'name' and 'messages' keys
            iterations_per_model: Number of times to run each prompt on each model
            
        Returns:
            DataFrame with all comparison results
        """
        print(f"🤖 Starting Model Comparison: {self.comparison_name}")
        print(f"📊 Models: {list(self.model_configs.keys())}")
        print(f"📝 Prompts: {len(test_prompts)}")
        print(f"🔄 Iterations per model: {iterations_per_model}")
        print("=" * 60)
        
        for prompt_data in test_prompts:
            prompt_name = prompt_data["name"]
            messages = prompt_data["messages"]
            
            print(f"\\n📝 Testing prompt: {prompt_name}")
            
            for model_name, model_config in self.model_configs.items():
                print(f"  🤖 Model: {model_name}")
                
                # Check if we have API access for this model
                if model_name != "gpt-4o-mini":
                    print(f"    ⚠️  Skipping {model_name} (API key only available for gpt-4o-mini)")
                    continue
                
                for iteration in range(iterations_per_model):
                    print(f"    ⏳ Iteration {iteration + 1}/{iterations_per_model}")
                    
                    try:
                        # Call the model
                        model_function = self.available_models[model_name]
                        response, metrics = model_function(model_name, messages, model_config)
                        
                        # Record result
                        result = {
                            'timestamp': datetime.now().isoformat(),
                            'comparison_name': self.comparison_name,
                            'prompt_name': prompt_name,
                            'model': model_name,
                            'iteration': iteration + 1,
                            'response': response,
                            'messages': json.dumps(messages),
                            'model_config': json.dumps(model_config),
                            **metrics
                        }
                        
                        self.results.append(result)
                        
                        print(f"      ✅ {metrics['response_length']} tokens, ${metrics['cost_usd']:.6f}, {metrics['latency_ms']:.0f}ms")
                        
                    except Exception as e:
                        print(f"      ❌ Error: {e}")
                        continue
        
        return pd.DataFrame(self.results)
    
    def analyze_model_performance(self) -> Dict[str, Any]:
        """
        Analyze performance differences between models.
        
        Returns:
            Comprehensive analysis dictionary
        """
        if not self.results:
            return {"error": "No results to analyze"}
        
        df = pd.DataFrame(self.results)
        
        analysis = {
            'comparison_name': self.comparison_name,
            'timestamp': datetime.now().isoformat(),
            'models_compared': df['model'].unique().tolist(),
            'prompts_tested': df['prompt_name'].unique().tolist(),
            'total_runs': len(self.results),
            'model_performance': {}
        }
        
        # Analyze each model's performance
        for model in df['model'].unique():
            model_data = df[df['model'] == model]
            
            model_stats = {
                'total_runs': len(model_data),
                'avg_response_length': float(model_data['response_length'].mean()),
                'avg_cost_per_response': float(model_data['cost_usd'].mean()),
                'avg_latency_ms': float(model_data['latency_ms'].mean()),
                'total_cost': float(model_data['cost_usd'].sum()),
                'cost_per_token': float(model_data['cost_usd'].sum() / model_data['total_tokens'].sum()) if model_data['total_tokens'].sum() > 0 else 0,
                'response_consistency': float(model_data['response_length'].std()),  # Lower = more consistent
            }
            
            # Calculate cost efficiency (tokens per dollar)
            if model_stats['total_cost'] > 0:
                model_stats['tokens_per_dollar'] = float(model_data['total_tokens'].sum() / model_stats['total_cost'])
            else:
                model_stats['tokens_per_dollar'] = 0
            
            analysis['model_performance'][model] = model_stats
        
        # Cross-model comparisons
        if len(analysis['models_compared']) > 1:
            analysis['comparisons'] = {}
            
            metrics_to_compare = ['avg_response_length', 'avg_cost_per_response', 'avg_latency_ms', 'tokens_per_dollar']
            
            for metric in metrics_to_compare:
                metric_comparison = {}
                for model in analysis['models_compared']:
                    metric_comparison[model] = analysis['model_performance'][model][metric]
                
                # Find best and worst
                best_model = max(metric_comparison.items(), key=lambda x: x[1] if 'tokens_per_dollar' in metric or 'response_length' in metric else -x[1])
                worst_model = min(metric_comparison.items(), key=lambda x: x[1] if 'tokens_per_dollar' in metric or 'response_length' in metric else -x[1])
                
                analysis['comparisons'][metric] = {
                    'values': metric_comparison,
                    'best': best_model[0],
                    'worst': worst_model[0],
                    'best_value': best_model[1],
                    'worst_value': worst_model[1]
                }
        
        return analysis
    
    def generate_comparison_report(self) -> str:
        """
        Generate a comprehensive model comparison report.
        
        Returns:
            Path to the generated report file
        """
        analysis = self.analyze_model_performance()
        
        if 'error' in analysis:
            return analysis['error']
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed analysis
        analysis_file = self.output_dir / f"{self.comparison_name}_{timestamp}_analysis.json"
        with open(analysis_file, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        # Save results CSV
        if self.results:
            df = pd.DataFrame(self.results)
            csv_file = self.output_dir / f"{self.comparison_name}_{timestamp}_results.csv"
            df.to_csv(csv_file, index=False)
        
        # Generate human-readable report
        report_file = self.output_dir / f"{self.comparison_name}_{timestamp}_report.md"
        
        with open(report_file, 'w') as f:
            f.write(f"# Model Comparison Report: {self.comparison_name}\\n\\n")
            f.write(f"**Generated:** {analysis['timestamp']}\\n")
            f.write(f"**Models:** {', '.join(analysis['models_compared'])}\\n")
            f.write(f"**Prompts:** {', '.join(analysis['prompts_tested'])}\\n")
            f.write(f"**Total Runs:** {analysis['total_runs']}\\n\\n")
            
            # Performance summary
            f.write("## Model Performance Summary\\n\\n")
            f.write("| Model | Avg Response Length | Avg Cost | Avg Latency | Tokens/$1 | Consistency |\\n")
            f.write("|-------|-------------------|----------|-------------|-----------|-------------|\\n")
            
            for model, stats in analysis['model_performance'].items():
                f.write(f"| {model} | {stats['avg_response_length']:.1f} | ${stats['avg_cost_per_response']:.6f} | {stats['avg_latency_ms']:.0f}ms | {stats['tokens_per_dollar']:.0f} | {stats['response_consistency']:.1f} |\\n")
            
            f.write("\\n")
            
            # Detailed metrics
            f.write("## Detailed Metrics\\n\\n")
            for model, stats in analysis['model_performance'].items():
                f.write(f"### {model}\\n\\n")
                f.write(f"- **Total Runs:** {stats['total_runs']}\\n")
                f.write(f"- **Average Response Length:** {stats['avg_response_length']:.1f} tokens\\n")
                f.write(f"- **Average Cost per Response:** ${stats['avg_cost_per_response']:.6f}\\n")
                f.write(f"- **Average Latency:** {stats['avg_latency_ms']:.0f}ms\\n")
                f.write(f"- **Total Cost:** ${stats['total_cost']:.6f}\\n")
                f.write(f"- **Cost per Token:** ${stats['cost_per_token']:.8f}\\n")
                f.write(f"- **Tokens per Dollar:** {stats['tokens_per_dollar']:.0f}\\n")
                f.write(f"- **Response Consistency:** {stats['response_consistency']:.1f} (std dev)\\n\\n")
            
            # Recommendations
            if len(analysis['models_compared']) > 1 and 'comparisons' in analysis:
                f.write("## Recommendations\\n\\n")
                
                comparisons = analysis['comparisons']
                
                if 'tokens_per_dollar' in comparisons:
                    best_value = comparisons['tokens_per_dollar']['best']
                    f.write(f"- **Most Cost-Efficient:** {best_value} ({comparisons['tokens_per_dollar']['best_value']:.0f} tokens/$1)\\n")
                
                if 'avg_latency_ms' in comparisons:
                    fastest = comparisons['avg_latency_ms']['best']  # 'best' is actually lowest latency
                    f.write(f"- **Fastest Response:** {fastest} ({comparisons['avg_latency_ms']['best_value']:.0f}ms average)\\n")
                
                if 'avg_response_length' in comparisons:
                    most_detailed = comparisons['avg_response_length']['best']
                    f.write(f"- **Most Detailed Responses:** {most_detailed} ({comparisons['avg_response_length']['best_value']:.1f} tokens average)\\n")
        
        print(f"🤖 Model Comparison Report Generated:")
        print(f"   📄 Analysis: {analysis_file}")
        print(f"   📈 Results: {csv_file}")
        print(f"   📝 Report: {report_file}")
        
        return str(report_file)

# Example usage function
def create_example_model_comparison():
    """
    Example of how to use the Model Comparison Framework.
    Currently set up for gpt-4o-mini only, but easily extensible.
    """
    print("🤖 Model Comparison Framework Example")
    print("=" * 45)
    
    # Create comparison
    comparison = ModelComparisonFramework("writing_style_comparison")
    
    # Add models (currently only gpt-4o-mini available)
    comparison.add_model_config("gpt-4o-mini", {"temperature": 0.7})
    
    # Future: Add more models when API keys are available
    # comparison.add_model_config("gpt-4o", {"temperature": 0.7})
    # comparison.add_model_config("gpt-4", {"temperature": 0.7})
    
    # Define test prompts
    test_prompts = [
        {
            "name": "creative_writing",
            "messages": [
                {"role": "system", "content": "You are a creative writing assistant."},
                {"role": "user", "content": "Write a short story opening about a mysterious door."}
            ]
        },
        {
            "name": "technical_explanation",
            "messages": [
                {"role": "system", "content": "You are a technical documentation expert."},
                {"role": "user", "content": "Explain how machine learning works in simple terms."}
            ]
        },
        {
            "name": "poetry_generation",
            "messages": [
                {"role": "system", "content": "You are a poetry expert."},
                {"role": "user", "content": "Write a haiku about the ocean at sunset."}
            ]
        }
    ]
    
    print(f"\\n📝 Test prompts prepared: {len(test_prompts)}")
    print("\\n🚀 Ready to run model comparison!")
    print("Use: comparison.run_comparison(test_prompts, iterations_per_model=2)")
    
    return comparison, test_prompts

if __name__ == "__main__":
    # Run example
    comp, prompts = create_example_model_comparison()
