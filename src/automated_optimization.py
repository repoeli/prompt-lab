"""
Automated Optimization Framework

This module provides automated prompt optimization using machine learning techniques
including genetic algorithms, gradient-free optimization, and performance-based ranking.

Key Features:
- Genetic Algorithm for prompt evolution
- Performance-based prompt ranking
- Automated parameter tuning
- ML-driven prompt improvement suggestions
- Cost-aware optimization strategies
"""

import random
import json
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional, Callable
from dataclasses import dataclass, asdict
from datetime import datetime
import itertools

# Lazy imports for ML libraries to avoid potential import issues
def _get_sklearn_components():
    """Lazy import for sklearn components to avoid startup issues"""
    try:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import mean_squared_error, r2_score
        return {
            'RandomForestRegressor': RandomForestRegressor,
            'train_test_split': train_test_split,
            'StandardScaler': StandardScaler,
            'mean_squared_error': mean_squared_error,
            'r2_score': r2_score
        }
    except ImportError:
        print("Warning: sklearn not available. ML-based optimization will be limited.")
        return None

@dataclass
class PromptCandidate:
    """Represents a prompt candidate with its performance metrics."""
    prompt: str
    system_message: str
    temperature: float
    max_tokens: int
    fitness_score: float = 0.0
    cost: float = 0.0
    response_time: float = 0.0
    generation: int = 0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class OptimizationConfig:
    """Configuration for optimization algorithms."""
    population_size: int = 20
    generations: int = 10
    mutation_rate: float = 0.1
    crossover_rate: float = 0.7
    elite_ratio: float = 0.2
    cost_weight: float = 0.3
    performance_weight: float = 0.7
    target_metrics: Dict[str, float] = None
    
    def __post_init__(self):
        if self.target_metrics is None:
            self.target_metrics = {
                'min_response_quality': 0.8,
                'max_cost_per_request': 0.01,
                'max_response_time': 5.0
            }

class GeneticOptimizer:
    """
    Genetic Algorithm for prompt optimization.
    
    This class evolves prompts over generations to find optimal combinations
    of prompt text, system messages, and model parameters.
    """
    
    def __init__(self, config: OptimizationConfig, fitness_function: Callable):
        self.config = config
        self.fitness_function = fitness_function
        self.population: List[PromptCandidate] = []
        self.generation_history: List[Dict] = []
        
        # Prompt building blocks for genetic operations
        self.prompt_templates = [
            "You are a {role}. {instruction}",
            "As a {role}, {instruction}",
            "{instruction} Please {style}.",
            "{role_description} {instruction}",
            "Your task is to {action}. {context}"
        ]
        
        self.role_variations = [
            "helpful assistant", "expert advisor", "professional consultant",
            "skilled specialist", "knowledgeable guide", "experienced professional"
        ]
        
        self.style_variations = [
            "be concise and clear", "provide detailed explanations",
            "be creative and engaging", "maintain a professional tone",
            "be thorough and analytical", "focus on practical solutions"
        ]
        
        self.instruction_modifiers = [
            "carefully", "thoughtfully", "systematically", "creatively",
            "analytically", "professionally", "efficiently", "thoroughly"
        ]
    
    def initialize_population(self, base_prompt: str, system_message: str) -> None:
        """Initialize the population with variations of the base prompt."""
        self.population = []
        
        for _ in range(self.config.population_size):
            candidate = self._create_random_candidate(base_prompt, system_message)
            self.population.append(candidate)
    
    def _create_random_candidate(self, base_prompt: str, system_message: str) -> PromptCandidate:
        """Create a random prompt candidate with variations."""
        # Generate prompt variations
        template = random.choice(self.prompt_templates)
        role = random.choice(self.role_variations)
        style = random.choice(self.style_variations)
        modifier = random.choice(self.instruction_modifiers)
        
        # Apply variations to create new prompt
        if "{role}" in template:
            varied_prompt = template.format(
                role=role,
                instruction=f"{modifier} {base_prompt}",
                style=style,
                role_description=f"You are a {role}",
                action=base_prompt.lower(),
                context="Focus on providing high-quality responses"
            )
        else:
            varied_prompt = f"{modifier.capitalize()} {base_prompt}"
        
        # Vary system message
        system_variations = [
            system_message,
            f"{system_message} Be {style}.",
            f"As a {role}, {system_message.lower()}",
            f"{system_message} {modifier.capitalize()} approach each request."
        ]
        varied_system = random.choice(system_variations)
        
        # Vary model parameters
        temperature = round(random.uniform(0.1, 1.0), 2)
        max_tokens = random.choice([100, 150, 200, 300, 500, 1000])
        
        return PromptCandidate(
            prompt=varied_prompt,
            system_message=varied_system,
            temperature=temperature,
            max_tokens=max_tokens,
            generation=0
        )
    
    def evaluate_population(self) -> None:
        """Evaluate fitness for all candidates in the population."""
        for candidate in self.population:
            if candidate.fitness_score == 0.0:  # Only evaluate if not already evaluated
                candidate.fitness_score = self.fitness_function(candidate)
    
    def selection(self) -> List[PromptCandidate]:
        """Select parents for reproduction using tournament selection."""
        parents = []
        tournament_size = max(2, self.config.population_size // 10)
        
        for _ in range(self.config.population_size):
            tournament = random.sample(self.population, tournament_size)
            winner = max(tournament, key=lambda x: x.fitness_score)
            parents.append(winner)
        
        return parents
    
    def crossover(self, parent1: PromptCandidate, parent2: PromptCandidate) -> Tuple[PromptCandidate, PromptCandidate]:
        """Create offspring by combining traits from two parents."""
        if random.random() > self.config.crossover_rate:
            return parent1, parent2
        
        # Combine prompts by taking elements from both
        prompt_parts1 = parent1.prompt.split()
        prompt_parts2 = parent2.prompt.split()
        
        # Create hybrid prompts
        crossover_point = random.randint(1, min(len(prompt_parts1), len(prompt_parts2)) - 1)
        child1_prompt = " ".join(prompt_parts1[:crossover_point] + prompt_parts2[crossover_point:])
        child2_prompt = " ".join(prompt_parts2[:crossover_point] + prompt_parts1[crossover_point:])
        
        # Inherit other parameters
        child1 = PromptCandidate(
            prompt=child1_prompt,
            system_message=parent2.system_message,  # Take from other parent
            temperature=(parent1.temperature + parent2.temperature) / 2,
            max_tokens=random.choice([parent1.max_tokens, parent2.max_tokens]),
            generation=max(parent1.generation, parent2.generation) + 1
        )
        
        child2 = PromptCandidate(
            prompt=child2_prompt,
            system_message=parent1.system_message,
            temperature=(parent1.temperature + parent2.temperature) / 2,
            max_tokens=random.choice([parent1.max_tokens, parent2.max_tokens]),
            generation=max(parent1.generation, parent2.generation) + 1
        )
        
        return child1, child2
    
    def mutate(self, candidate: PromptCandidate) -> PromptCandidate:
        """Apply mutations to a candidate."""
        if random.random() > self.config.mutation_rate:
            return candidate
        
        # Choose what to mutate
        mutation_type = random.choice(['prompt', 'system', 'temperature', 'max_tokens'])
        
        if mutation_type == 'prompt':
            # Add a modifier or rephrase part of the prompt
            modifier = random.choice(self.instruction_modifiers)
            candidate.prompt = f"{modifier.capitalize()} {candidate.prompt}"
        
        elif mutation_type == 'system':
            # Modify system message
            style = random.choice(self.style_variations)
            candidate.system_message += f" {style.capitalize()}."
        
        elif mutation_type == 'temperature':
            # Adjust temperature slightly
            candidate.temperature = max(0.1, min(1.0, 
                candidate.temperature + random.uniform(-0.2, 0.2)))
        
        elif mutation_type == 'max_tokens':
            # Adjust max_tokens
            tokens_options = [100, 150, 200, 300, 500, 1000]
            candidate.max_tokens = random.choice(tokens_options)
        
        return candidate
    
    def evolve_generation(self) -> None:
        """Evolve the population by one generation."""
        # Evaluate current population
        self.evaluate_population()
        
        # Keep elite candidates
        elite_count = int(self.config.population_size * self.config.elite_ratio)
        elite = sorted(self.population, key=lambda x: x.fitness_score, reverse=True)[:elite_count]
        
        # Select parents and create offspring
        parents = self.selection()
        offspring = []
        
        for i in range(0, len(parents) - 1, 2):
            child1, child2 = self.crossover(parents[i], parents[i + 1])
            child1 = self.mutate(child1)
            child2 = self.mutate(child2)
            offspring.extend([child1, child2])
        
        # Create new population
        new_population = elite + offspring[:self.config.population_size - elite_count]
        self.population = new_population[:self.config.population_size]
        
        # Record generation statistics
        current_gen = max([c.generation for c in self.population])
        avg_fitness = np.mean([c.fitness_score for c in self.population])
        best_fitness = max([c.fitness_score for c in self.population])
        
        self.generation_history.append({
            'generation': current_gen,
            'avg_fitness': avg_fitness,
            'best_fitness': best_fitness,
            'population_size': len(self.population)
        })
    
    def optimize(self, base_prompt: str, system_message: str) -> PromptCandidate:
        """Run the genetic algorithm optimization."""
        print(f"🧬 Starting Genetic Algorithm Optimization")
        print(f"Population: {self.config.population_size}, Generations: {self.config.generations}")
        
        # Initialize population
        self.initialize_population(base_prompt, system_message)
        
        # Evolve over generations
        for generation in range(self.config.generations):
            print(f"Generation {generation + 1}/{self.config.generations}...")
            self.evolve_generation()
            
            if self.generation_history:
                latest = self.generation_history[-1]
                print(f"  Best fitness: {latest['best_fitness']:.3f}, Avg: {latest['avg_fitness']:.3f}")
        
        # Return best candidate
        best_candidate = max(self.population, key=lambda x: x.fitness_score)
        print(f"🎯 Optimization complete! Best fitness: {best_candidate.fitness_score:.3f}")
        
        return best_candidate


class PerformancePredictor:
    """
    Machine learning model to predict prompt performance.
    
    Uses historical data to predict how well a prompt will perform
    before actually running it through the API.    """
    
    def __init__(self):
        sklearn_components = _get_sklearn_components()
        if sklearn_components:
            self.model = sklearn_components['RandomForestRegressor'](n_estimators=100, random_state=42)
            self.scaler = sklearn_components['StandardScaler']()
        else:
            self.model = None
            self.scaler = None
        self.feature_columns = []
        self.is_trained = False
    
    def extract_features(self, candidate: PromptCandidate) -> Dict[str, float]:
        """Extract numerical features from a prompt candidate."""
        prompt_text = candidate.prompt + " " + candidate.system_message
        
        features = {
            'prompt_length': len(prompt_text),
            'word_count': len(prompt_text.split()),
            'sentence_count': prompt_text.count('.') + prompt_text.count('!') + prompt_text.count('?'),
            'temperature': candidate.temperature,
            'max_tokens': candidate.max_tokens,
            'exclamation_count': prompt_text.count('!'),
            'question_count': prompt_text.count('?'),
            'uppercase_ratio': sum(1 for c in prompt_text if c.isupper()) / len(prompt_text) if prompt_text else 0,
            'avg_word_length': np.mean([len(word) for word in prompt_text.split()]) if prompt_text.split() else 0,
            'contains_please': 1 if 'please' in prompt_text.lower() else 0,
            'contains_help': 1 if 'help' in prompt_text.lower() else 0,
            'contains_specific': 1 if any(word in prompt_text.lower() for word in ['specific', 'detailed', 'exact']) else 0,
        }
        
        return features
    
    def train(self, historical_data: List[PromptCandidate]) -> Dict[str, float]:
        """Train the performance prediction model."""
        if len(historical_data) < 10:
            raise ValueError("Need at least 10 historical samples to train the model")
        
        # Extract features and targets
        feature_dicts = [self.extract_features(candidate) for candidate in historical_data]
        features_df = pd.DataFrame(feature_dicts)
        targets = np.array([candidate.fitness_score for candidate in historical_data])
        
        # Store feature columns for consistency
        self.feature_columns = features_df.columns.tolist()
        
        # Scale features
        features_scaled = self.scaler.fit_transform(features_df)
          # Split data
        sklearn_components = _get_sklearn_components()
        if not sklearn_components:
            return {"error": "sklearn not available"}
            
        X_train, X_test, y_train, y_test = sklearn_components['train_test_split'](
            features_scaled, targets, test_size=0.2, random_state=42
        )
        
        # Train model
        self.model.fit(X_train, y_train)
          # Evaluate
        y_pred = self.model.predict(X_test)
        mse = sklearn_components['mean_squared_error'](y_test, y_pred)
        r2 = sklearn_components['r2_score'](y_test, y_pred)
        
        self.is_trained = True
        
        return {
            'mse': mse,
            'r2_score': r2,
            'training_samples': len(historical_data),
            'feature_importance': dict(zip(self.feature_columns, self.model.feature_importances_))
        }
    
    def predict_performance(self, candidate: PromptCandidate) -> float:
        """Predict performance score for a candidate."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        features = self.extract_features(candidate)
        features_df = pd.DataFrame([features])[self.feature_columns]
        features_scaled = self.scaler.transform(features_df)
        
        prediction = self.model.predict(features_scaled)[0]
        return float(prediction)


class AutomatedOptimizer:
    """
    Main class for automated prompt optimization.
    
    Combines genetic algorithms, performance prediction, and cost optimization
    to automatically improve prompts over time.
    """
    
    def __init__(self, config: OptimizationConfig = None):
        self.config = config or OptimizationConfig()
        self.predictor = PerformancePredictor()
        self.optimization_history: List[Dict] = []
        self.best_candidates: List[PromptCandidate] = []
    
    def fitness_function(self, candidate: PromptCandidate) -> float:
        """
        Calculate fitness score for a candidate.
        
        This is a placeholder - in practice, you would run the prompt
        through your evaluation framework and calculate a real score.
        """
        # Simulate evaluation (replace with actual evaluation)
        base_score = random.uniform(0.5, 1.0)
        
        # Penalize for cost and response time
        cost_penalty = candidate.cost * self.config.cost_weight if candidate.cost > 0 else 0
        time_penalty = candidate.response_time * 0.01 if candidate.response_time > 0 else 0
        
        # Prefer certain characteristics
        prompt_bonus = 0
        if len(candidate.prompt.split()) > 5:  # Prefer more detailed prompts
            prompt_bonus += 0.1
        if 'please' in candidate.prompt.lower():  # Prefer polite prompts
            prompt_bonus += 0.05
        
        final_score = base_score + prompt_bonus - cost_penalty - time_penalty
        return max(0.0, min(1.0, final_score))
    
    def optimize_prompt(self, base_prompt: str, system_message: str = "You are a helpful assistant.") -> Dict[str, Any]:
        """Run automated optimization on a prompt."""
        print(f"🚀 Starting Automated Prompt Optimization")
        print(f"Base prompt: '{base_prompt[:50]}...'")
        
        # Initialize genetic optimizer
        genetic_optimizer = GeneticOptimizer(self.config, self.fitness_function)
        
        # Run genetic algorithm
        start_time = datetime.now()
        best_candidate = genetic_optimizer.optimize(base_prompt, system_message)
        optimization_time = (datetime.now() - start_time).total_seconds()
        
        # Store results
        optimization_result = {
            'original_prompt': base_prompt,
            'original_system': system_message,
            'optimized_candidate': best_candidate,
            'optimization_time': optimization_time,
            'generations_history': genetic_optimizer.generation_history,
            'improvement_ratio': best_candidate.fitness_score / 0.7,  # Assume 0.7 baseline
            'timestamp': datetime.now().isoformat()
        }
        
        self.optimization_history.append(optimization_result)
        self.best_candidates.append(best_candidate)
        
        return optimization_result
    
    def batch_optimize(self, prompts: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """Optimize multiple prompts in batch."""
        results = []
        
        print(f"🔄 Batch optimizing {len(prompts)} prompts...")
        
        for i, prompt_data in enumerate(prompts, 1):
            print(f"\nOptimizing prompt {i}/{len(prompts)}")
            result = self.optimize_prompt(
                prompt_data.get('prompt', ''),
                prompt_data.get('system_message', 'You are a helpful assistant.')
            )
            results.append(result)
        
        return results
    
    def generate_optimization_report(self) -> str:
        """Generate a comprehensive optimization report."""
        if not self.optimization_history:
            return "No optimization runs to report."
        
        report_lines = [
            "# 🤖 Automated Optimization Report",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Summary",
            f"- Total optimization runs: {len(self.optimization_history)}",
            f"- Average improvement ratio: {np.mean([r['improvement_ratio'] for r in self.optimization_history]):.2f}x",
            f"- Total optimization time: {sum([r['optimization_time'] for r in self.optimization_history]):.1f} seconds",
            "",
            "## Best Candidates",
        ]
        
        # Sort by fitness score
        top_candidates = sorted(self.best_candidates, key=lambda x: x.fitness_score, reverse=True)[:5]
        
        for i, candidate in enumerate(top_candidates, 1):
            report_lines.extend([
                f"### {i}. Fitness Score: {candidate.fitness_score:.3f}",
                f"**Prompt:** {candidate.prompt}",
                f"**System:** {candidate.system_message}",
                f"**Parameters:** T={candidate.temperature}, Max tokens={candidate.max_tokens}",
                f"**Generation:** {candidate.generation}",
                ""
            ])
        
        # Add optimization trends
        if self.optimization_history:
            avg_generations = np.mean([len(r['generations_history']) for r in self.optimization_history])
            report_lines.extend([
                "## Optimization Trends",
                f"- Average generations per run: {avg_generations:.1f}",
                f"- Success rate: {len([r for r in self.optimization_history if r['improvement_ratio'] > 1.0]) / len(self.optimization_history) * 100:.1f}%",
                ""
            ])
        
        # Add recommendations
        report_lines.extend([
            "## Recommendations",
            "Based on the optimization runs, consider:",
            "- Adjusting mutation rates if improvement plateaus early",
            "- Increasing population size for more diverse exploration",
            "- Fine-tuning the fitness function weights",
            "- Adding domain-specific prompt templates",
        ])
        
        return "\n".join(report_lines)
    
    def save_results(self, filepath: str) -> None:
        """Save optimization results to JSON file."""
        results_data = {
            'config': asdict(self.config),
            'optimization_history': [
                {
                    **result,
                    'optimized_candidate': asdict(result['optimized_candidate'])
                }
                for result in self.optimization_history
            ],
            'best_candidates': [asdict(candidate) for candidate in self.best_candidates],
            'summary': {
                'total_runs': len(self.optimization_history),
                'avg_improvement': np.mean([r['improvement_ratio'] for r in self.optimization_history]) if self.optimization_history else 0,
                'generated_at': datetime.now().isoformat()
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        print(f"💾 Optimization results saved to {filepath}")


# Example usage and testing functions
def create_sample_fitness_function():
    """Create a sample fitness function for testing."""
    def fitness_fn(candidate: PromptCandidate) -> float:
        # Simple fitness based on prompt characteristics
        score = 0.5  # Base score
        
        # Reward longer, more detailed prompts
        word_count = len(candidate.prompt.split())
        if word_count > 10:
            score += 0.2
        if word_count > 20:
            score += 0.1
        
        # Reward polite language
        if 'please' in candidate.prompt.lower():
            score += 0.1
        
        # Penalize extreme temperatures
        if candidate.temperature < 0.3 or candidate.temperature > 0.8:
            score -= 0.1
        
        # Add some randomness to simulate real evaluation variance
        score += random.uniform(-0.1, 0.1)
        
        return max(0.0, min(1.0, score))
    
    return fitness_fn


if __name__ == "__main__":
    # Example usage
    print("🧪 Testing Automated Optimization Framework")
    
    # Create optimizer
    config = OptimizationConfig(
        population_size=10,
        generations=5,
        mutation_rate=0.2
    )
    optimizer = AutomatedOptimizer(config)
    
    # Test optimization
    result = optimizer.optimize_prompt(
        "Help me write a professional email",
        "You are a writing assistant."
    )
    
    print(f"\n✅ Optimization completed!")
    print(f"Best candidate: {result['optimized_candidate'].prompt}")
    print(f"Fitness score: {result['optimized_candidate'].fitness_score:.3f}")
    
    # Generate report
    print("\n📊 Optimization Report:")
    print(optimizer.generate_optimization_report())
