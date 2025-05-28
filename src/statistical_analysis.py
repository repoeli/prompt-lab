"""
Statistical Analysis Tools for Prompt Engineering
=================================================

This module provides advanced statistical analysis capabilities for
prompt engineering experiments and A/B tests.

Key Features:
- Statistical significance testing
- Effect size calculations  
- Confidence intervals
- Trend analysis
- Performance correlations
- Cost efficiency analysis
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

# Lazy imports for plotting libraries to avoid potential import issues
def _get_scipy_stats():
    """Lazy import for scipy.stats to avoid startup issues"""
    try:
        from scipy import stats
        return stats
    except ImportError:
        print("Warning: scipy not available. Statistical tests will be limited.")
        return None

def _get_matplotlib():
    """Lazy import for matplotlib to avoid startup issues"""
    try:
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        print("Warning: matplotlib not available. Plotting will be limited.")
        return None

def _get_seaborn():
    """Lazy import for seaborn to avoid startup issues"""
    try:
        import seaborn as sns
        return sns
    except ImportError:
        print("Warning: seaborn not available. Advanced plotting will be limited.")
        return None

class StatisticalAnalyzer:
    """
    Comprehensive statistical analysis for prompt engineering experiments.
    
    This class provides:
    - Descriptive statistics
    - Inferential statistics (t-tests, ANOVA, etc.)
    - Effect size calculations
    - Trend analysis over time
    - Cost-performance relationships
    """
    
    def __init__(self, analysis_name: str, output_dir: str = None):
        self.analysis_name = analysis_name
        if output_dir is None:
            # Get the project root directory (where this script is run from)
            project_root = Path(__file__).parent.parent
            output_dir = project_root / "data" / "statistical_analysis"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.data = None
        self.results = {}
    
    def load_ledger_data(self, ledger_file: str = None) -> pd.DataFrame:
        """
        Load and prepare data from the token ledger for analysis.
        
        Args:
            ledger_file: Path to the token ledger CSV file
            
        Returns:
            Processed DataFrame ready for analysis
        """
        if ledger_file is None:
            # Get the project root directory and default ledger path
            project_root = Path(__file__).parent.parent
            ledger_file = project_root / "data" / "token_ledger.csv"
        
        try:
            df = pd.read_csv(ledger_file)
            
            # Convert date column to datetime
            df['date'] = pd.to_datetime(df['date'])
            
            # Add derived columns for analysis
            df['total_tokens'] = df['tokens_in'] + df['tokens_out']
            df['cost_per_token'] = df['cost_usd'] / df['total_tokens'].replace(0, np.nan)
            df['tokens_per_dollar'] = df['total_tokens'] / df['cost_usd'].replace(0, np.nan)
            df['input_output_ratio'] = df['tokens_in'] / df['tokens_out'].replace(0, np.nan)
            df['hour'] = df['date'].dt.hour
            df['day_of_week'] = df['date'].dt.day_name()
            
            # Add time-based features
            df['days_since_start'] = (df['date'] - df['date'].min()).dt.days
            
            self.data = df
            print(f"✅ Loaded {len(df)} records from ledger")
            print(f"📅 Date range: {df['date'].min()} to {df['date'].max()}")
            print(f"🏷️  Phases: {df['phase'].unique().tolist()}")
            print(f"🤖 Models: {df['model'].unique().tolist()}")
            
            return df
            
        except Exception as e:
            print(f"❌ Error loading ledger data: {e}")
            return pd.DataFrame()
    
    def descriptive_statistics(self, group_by: Optional[str] = None) -> Dict[str, Any]:
        """
        Calculate comprehensive descriptive statistics.
        
        Args:
            group_by: Optional column to group by (e.g., 'phase', 'model')
            
        Returns:
            Dictionary with descriptive statistics
        """
        if self.data is None or len(self.data) == 0:
            return {"error": "No data loaded"}
        
        metrics = ['tokens_in', 'tokens_out', 'total_tokens', 'cost_usd', 'cost_per_token', 'tokens_per_dollar']
        
        if group_by:
            # Grouped statistics
            stats_dict = {}
            for group in self.data[group_by].unique():
                group_data = self.data[self.data[group_by] == group]
                stats_dict[str(group)] = {}
                
                for metric in metrics:
                    if metric in group_data.columns and not group_data[metric].isna().all():
                        stats_dict[str(group)][metric] = {
                            'count': int(group_data[metric].count()),
                            'mean': float(group_data[metric].mean()),
                            'median': float(group_data[metric].median()),
                            'std': float(group_data[metric].std()),
                            'min': float(group_data[metric].min()),
                            'max': float(group_data[metric].max()),                            'q25': float(group_data[metric].quantile(0.25)),
                            'q75': float(group_data[metric].quantile(0.75)),
                        }
                        
                        # Add scipy-based statistics if available
                        stats = _get_scipy_stats()
                        if stats:
                            stats_dict[str(group)][metric]['skewness'] = float(stats.skew(group_data[metric].dropna()))
                            stats_dict[str(group)][metric]['kurtosis'] = float(stats.kurtosis(group_data[metric].dropna()))
        else:
            # Overall statistics
            stats_dict = {}
            for metric in metrics:
                if metric in self.data.columns and not self.data[metric].isna().all():
                    stats_dict[metric] = {
                        'count': int(self.data[metric].count()),
                        'mean': float(self.data[metric].mean()),
                        'median': float(self.data[metric].median()),
                        'std': float(self.data[metric].std()),
                        'min': float(self.data[metric].min()),
                        'max': float(self.data[metric].max()),                        'q25': float(self.data[metric].quantile(0.25)),
                        'q75': float(self.data[metric].quantile(0.75)),
                    }
                    
                    # Add scipy-based statistics if available
                    stats = _get_scipy_stats()
                    if stats:
                        stats_dict[metric]['skewness'] = float(stats.skew(self.data[metric].dropna()))
                        stats_dict[metric]['kurtosis'] = float(stats.kurtosis(self.data[metric].dropna()))
        
        self.results['descriptive_stats'] = {
            'grouped_by': group_by,
            'timestamp': datetime.now().isoformat(),
            'statistics': stats_dict
        }
        
        return self.results['descriptive_stats']
    
    def hypothesis_testing(self, metric: str, group_column: str, alpha: float = 0.05) -> Dict[str, Any]:
        """
        Perform hypothesis testing between groups.
        
        Args:
            metric: Metric to test (e.g., 'cost_usd', 'tokens_out')
            group_column: Column to group by (e.g., 'phase', 'model')
            alpha: Significance level (default 0.05)
            
        Returns:
            Dictionary with test results
        """
        if self.data is None or len(self.data) == 0:
            return {"error": "No data loaded"}
        
        if metric not in self.data.columns:
            return {"error": f"Metric '{metric}' not found in data"}
        
        if group_column not in self.data.columns:
            return {"error": f"Group column '{group_column}' not found in data"}
        
        # Get unique groups
        groups = self.data[group_column].unique()
        group_data = [self.data[self.data[group_column] == group][metric].dropna() for group in groups]
        
        # Remove empty groups
        groups = [groups[i] for i, data in enumerate(group_data) if len(data) > 0]
        group_data = [data for data in group_data if len(data) > 0]
        
        if len(groups) < 2:
            return {"error": "Need at least 2 groups for comparison"}
        
        results = {
            'metric': metric,
            'group_column': group_column,
            'groups': [str(g) for g in groups],
            'alpha': alpha,
            'timestamp': datetime.now().isoformat()
        }
        
        # Descriptive statistics for each group
        results['group_stats'] = {}
        for i, group in enumerate(groups):
            data = group_data[i]
            results['group_stats'][str(group)] = {
                'count': len(data),
                'mean': float(data.mean()),
                'std': float(data.std()),
                'median': float(data.median())
            }
        
        # Choose appropriate test based on number of groups        if len(groups) == 2:
            # Two-sample t-test
            data1, data2 = group_data[0], group_data[1]
            
            stats = _get_scipy_stats()
            if stats:
                # Check for equal variances (Levene's test)
                levene_stat, levene_p = stats.levene(data1, data2)
                equal_var = levene_p > alpha
                
                # Perform t-test
                t_stat, p_value = stats.ttest_ind(data1, data2, equal_var=equal_var)
            else:
                # Fallback to simple comparison without statistical tests
                levene_stat, levene_p = None, None
                equal_var = True
                t_stat, p_value = None, None
            
            # Effect size (Cohen's d)
            pooled_std = np.sqrt(((len(data1) - 1) * data1.var() + 
                                (len(data2) - 1) * data2.var()) / 
                               (len(data1) + len(data2) - 2))
            cohens_d = (data1.mean() - data2.mean()) / pooled_std
            
            results['test'] = 'two_sample_t_test'
            results['equal_variances'] = equal_var
            results['levene_test'] = {'statistic': float(levene_stat), 'p_value': float(levene_p)}
            results['t_statistic'] = float(t_stat)
            results['p_value'] = float(p_value)
            results['significant'] = p_value < alpha
            results['cohens_d'] = float(cohens_d)
            results['effect_size'] = 'small' if abs(cohens_d) < 0.5 else 'medium' if abs(cohens_d) < 0.8 else 'large'
            
        else:
            # One-way ANOVA for multiple groups
            f_stat, p_value = stats.f_oneway(*group_data)
            
            results['test'] = 'one_way_anova'
            results['f_statistic'] = float(f_stat)
            results['p_value'] = float(p_value)
            results['significant'] = p_value < alpha
            
            # Effect size (eta-squared)
            ss_between = sum(len(data) * (data.mean() - self.data[metric].mean())**2 for data in group_data)
            ss_total = ((self.data[metric] - self.data[metric].mean())**2).sum()
            eta_squared = ss_between / ss_total
            results['eta_squared'] = float(eta_squared)
            
            # Post-hoc analysis if significant
            if p_value < alpha and len(groups) > 2:
                from scipy.stats import tukey_hsd
                pairwise_results = {}
                
                # Pairwise t-tests with Bonferroni correction
                from itertools import combinations
                n_comparisons = len(list(combinations(range(len(groups)), 2)))
                bonferroni_alpha = alpha / n_comparisons
                
                for i, j in combinations(range(len(groups)), 2):
                    group_i, group_j = str(groups[i]), str(groups[j])
                    data_i, data_j = group_data[i], group_data[j]
                    
                    t_stat, p_val = stats.ttest_ind(data_i, data_j)
                    
                    pairwise_results[f"{group_i}_vs_{group_j}"] = {
                        't_statistic': float(t_stat),
                        'p_value': float(p_val),
                        'significant_bonferroni': p_val < bonferroni_alpha,
                        'mean_difference': float(data_i.mean() - data_j.mean())
                    }
                
                results['pairwise_comparisons'] = pairwise_results
                results['bonferroni_alpha'] = bonferroni_alpha
        
        self.results['hypothesis_testing'] = results
        return results
    
    def trend_analysis(self, metric: str, time_window: str = '1D') -> Dict[str, Any]:
        """
        Analyze trends over time.
        
        Args:
            metric: Metric to analyze trends for
            time_window: Resampling window ('1D', '1W', '1M', etc.)
            
        Returns:
            Dictionary with trend analysis results
        """
        if self.data is None or len(self.data) == 0:
            return {"error": "No data loaded"}
        
        if metric not in self.data.columns:
            return {"error": f"Metric '{metric}' not found in data"}
        
        # Ensure data is sorted by date
        df_sorted = self.data.sort_values('date')
        
        # Resample data by time window
        df_resampled = df_sorted.set_index('date')[metric].resample(time_window).agg(['mean', 'count', 'std']).dropna()
        
        if len(df_resampled) < 2:
            return {"error": "Insufficient data for trend analysis"}
          # Calculate trend using linear regression
        x = np.arange(len(df_resampled))
        y = df_resampled['mean'].values
        
        stats = _get_scipy_stats()
        if stats:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        else:
            # Fallback simple trend calculation
            slope = (y[-1] - y[0]) / (len(y) - 1) if len(y) > 1 else 0
            intercept, r_value, p_value, std_err = 0, 0, 1, 0
        
        # Trend direction
        if abs(slope) < std_err:
            trend_direction = 'stable'
        elif slope > 0:
            trend_direction = 'increasing'
        else:
            trend_direction = 'decreasing'
        
        # Calculate percentage change
        if len(y) > 1:
            pct_change = ((y[-1] - y[0]) / y[0]) * 100 if y[0] != 0 else 0
        else:
            pct_change = 0
        
        results = {
            'metric': metric,
            'time_window': time_window,
            'timestamp': datetime.now().isoformat(),
            'data_points': len(df_resampled),
            'time_range': {
                'start': df_resampled.index[0].isoformat(),
                'end': df_resampled.index[-1].isoformat()
            },
            'trend': {
                'direction': trend_direction,
                'slope': float(slope),
                'r_squared': float(r_value**2),
                'p_value': float(p_value),
                'significant': p_value < 0.05,
                'percentage_change': float(pct_change)
            },
            'summary_stats': {
                'mean': float(df_resampled['mean'].mean()),
                'std': float(df_resampled['mean'].std()),
                'min': float(df_resampled['mean'].min()),
                'max': float(df_resampled['mean'].max())
            }
        }
        
        self.results['trend_analysis'] = results
        return results
    
    def correlation_analysis(self, metrics: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Analyze correlations between metrics.
        
        Args:
            metrics: List of metrics to include in correlation analysis
            
        Returns:
            Dictionary with correlation results
        """
        if self.data is None or len(self.data) == 0:
            return {"error": "No data loaded"}
        
        if metrics is None:
            metrics = ['tokens_in', 'tokens_out', 'total_tokens', 'cost_usd', 'cost_per_token', 'tokens_per_dollar']
        
        # Filter to only numeric columns that exist
        available_metrics = [m for m in metrics if m in self.data.columns and self.data[m].dtype in ['int64', 'float64']]
        
        if len(available_metrics) < 2:
            return {"error": "Need at least 2 numeric metrics for correlation analysis"}
        
        # Calculate correlation matrix
        corr_matrix = self.data[available_metrics].corr()
        
        # Convert to dictionary format
        correlations = {}
        for i, metric1 in enumerate(available_metrics):
            for j, metric2 in enumerate(available_metrics):
                if i < j:  # Only include upper triangle (avoid duplicates)
                    correlation = corr_matrix.loc[metric1, metric2]
                    
                    # Calculate significance
                    from scipy.stats import pearsonr
                    _, p_value = pearsonr(self.data[metric1].dropna(), self.data[metric2].dropna())
                    
                    # Interpret correlation strength
                    abs_corr = abs(correlation)
                    if abs_corr < 0.3:
                        strength = 'weak'
                    elif abs_corr < 0.7:
                        strength = 'moderate'
                    else:
                        strength = 'strong'
                    
                    correlations[f"{metric1}_vs_{metric2}"] = {
                        'correlation': float(correlation),
                        'p_value': float(p_value),
                        'significant': p_value < 0.05,
                        'strength': strength,
                        'direction': 'positive' if correlation > 0 else 'negative'
                    }
        
        results = {
            'metrics_analyzed': available_metrics,
            'timestamp': datetime.now().isoformat(),
            'correlations': correlations,
            'correlation_matrix': corr_matrix.to_dict()
        }
        
        self.results['correlation_analysis'] = results
        return results
    
    def cost_efficiency_analysis(self) -> Dict[str, Any]:
        """
        Analyze cost efficiency patterns and optimization opportunities.
        
        Returns:
            Dictionary with cost efficiency analysis
        """
        if self.data is None or len(self.data) == 0:
            return {"error": "No data loaded"}
        
        # Calculate efficiency metrics
        df = self.data.copy()
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'total_experiments': len(df),
            'cost_summary': {
                'total_cost': float(df['cost_usd'].sum()),
                'average_cost_per_experiment': float(df['cost_usd'].mean()),
                'cost_std': float(df['cost_usd'].std()),
                'most_expensive': float(df['cost_usd'].max()),
                'least_expensive': float(df['cost_usd'].min())
            },
            'efficiency_metrics': {
                'avg_tokens_per_dollar': float(df['tokens_per_dollar'].mean()),
                'avg_cost_per_token': float(df['cost_per_token'].mean()),
                'input_output_ratio': float(df['input_output_ratio'].mean())
            }
        }
        
        # Efficiency by phase/model
        if 'phase' in df.columns:
            phase_efficiency = {}
            for phase in df['phase'].unique():
                phase_data = df[df['phase'] == phase]
                phase_efficiency[str(phase)] = {
                    'total_cost': float(phase_data['cost_usd'].sum()),
                    'avg_cost': float(phase_data['cost_usd'].mean()),
                    'tokens_per_dollar': float(phase_data['tokens_per_dollar'].mean()),
                    'experiments': len(phase_data)
                }
            results['efficiency_by_phase'] = phase_efficiency
        
        if 'model' in df.columns:
            model_efficiency = {}
            for model in df['model'].unique():
                model_data = df[df['model'] == model]
                model_efficiency[str(model)] = {
                    'total_cost': float(model_data['cost_usd'].sum()),
                    'avg_cost': float(model_data['cost_usd'].mean()),
                    'tokens_per_dollar': float(model_data['tokens_per_dollar'].mean()),
                    'experiments': len(model_data)
                }
            results['efficiency_by_model'] = model_efficiency
        
        # Identify optimization opportunities
        optimization_tips = []
        
        # Check for high-cost outliers
        q75 = df['cost_usd'].quantile(0.75)
        q25 = df['cost_usd'].quantile(0.25)
        iqr = q75 - q25
        outlier_threshold = q75 + 1.5 * iqr
        outliers = df[df['cost_usd'] > outlier_threshold]
        
        if len(outliers) > 0:
            optimization_tips.append(f"Found {len(outliers)} high-cost experiments (>${outlier_threshold:.6f}+). Review for optimization.")
        
        # Check input/output ratio efficiency
        avg_ratio = df['input_output_ratio'].mean()
        if avg_ratio > 2:
            optimization_tips.append("High input/output token ratio suggests prompts could be more concise.")
        elif avg_ratio < 0.5:
            optimization_tips.append("Low input/output ratio suggests responses could be more detailed.")
        
        # Check for cost trends
        if len(df) > 5:
            recent_cost = df.tail(5)['cost_usd'].mean()
            older_cost = df.head(5)['cost_usd'].mean()
            if recent_cost > older_cost * 1.2:
                optimization_tips.append("Recent experiments are 20%+ more expensive than earlier ones.")
        
        results['optimization_opportunities'] = optimization_tips
        
        self.results['cost_efficiency'] = results
        return results
    
    def generate_analysis_report(self) -> str:
        """
        Generate a comprehensive statistical analysis report.
        
        Returns:
            Path to the generated report file
        """
        if not self.results:
            return "No analysis results available. Run analysis methods first."
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results
        results_file = self.output_dir / f"{self.analysis_name}_{timestamp}_results.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Generate human-readable report
        report_file = self.output_dir / f"{self.analysis_name}_{timestamp}_report.md"
        
        with open(report_file, 'w') as f:
            f.write(f"# Statistical Analysis Report: {self.analysis_name}\\n\\n")
            f.write(f"**Generated:** {datetime.now().isoformat()}\\n\\n")
            
            # Data overview
            if self.data is not None:
                f.write(f"## Data Overview\\n\\n")
                f.write(f"- **Total Records:** {len(self.data)}\\n")
                f.write(f"- **Date Range:** {self.data['date'].min()} to {self.data['date'].max()}\\n")
                f.write(f"- **Models:** {', '.join(self.data['model'].unique())}\\n")
                f.write(f"- **Phases:** {', '.join(self.data['phase'].unique())}\\n\\n")
            
            # Include results from each analysis
            if 'descriptive_stats' in self.results:
                f.write("## Descriptive Statistics\\n\\n")
                stats = self.results['descriptive_stats']['statistics']
                if isinstance(stats, dict) and 'cost_usd' in stats:
                    f.write(f"### Cost Analysis\\n")
                    cost_stats = stats['cost_usd']
                    f.write(f"- **Mean:** ${cost_stats['mean']:.6f}\\n")
                    f.write(f"- **Median:** ${cost_stats['median']:.6f}\\n")
                    f.write(f"- **Range:** ${cost_stats['min']:.6f} - ${cost_stats['max']:.6f}\\n\\n")
            
            if 'hypothesis_testing' in self.results:
                f.write("## Hypothesis Testing\\n\\n")
                test = self.results['hypothesis_testing']
                f.write(f"**Test:** {test.get('test', 'Unknown')}\\n")
                f.write(f"**Metric:** {test.get('metric', 'Unknown')}\\n")
                f.write(f"**P-value:** {test.get('p_value', 'Unknown')}\\n")
                f.write(f"**Significant:** {'Yes' if test.get('significant', False) else 'No'}\\n\\n")
            
            if 'trend_analysis' in self.results:
                f.write("## Trend Analysis\\n\\n")
                trend = self.results['trend_analysis']
                f.write(f"**Metric:** {trend.get('metric', 'Unknown')}\\n")
                f.write(f"**Direction:** {trend.get('trend', {}).get('direction', 'Unknown')}\\n")
                f.write(f"**Percentage Change:** {trend.get('trend', {}).get('percentage_change', 0):.1f}%\\n\\n")
            
            if 'cost_efficiency' in self.results:
                f.write("## Cost Efficiency Analysis\\n\\n")
                cost = self.results['cost_efficiency']
                f.write(f"**Total Cost:** ${cost.get('cost_summary', {}).get('total_cost', 0):.6f}\\n")
                f.write(f"**Average per Experiment:** ${cost.get('cost_summary', {}).get('average_cost_per_experiment', 0):.6f}\\n")
                f.write(f"**Tokens per Dollar:** {cost.get('efficiency_metrics', {}).get('avg_tokens_per_dollar', 0):.0f}\\n\\n")
                
                if 'optimization_opportunities' in cost:
                    f.write("### Optimization Opportunities\\n\\n")
                    for tip in cost['optimization_opportunities']:
                        f.write(f"- {tip}\\n")
                    f.write("\\n")
        
        print(f"📊 Statistical Analysis Report Generated:")
        print(f"   📄 Results: {results_file}")
        print(f"   📝 Report: {report_file}")
        
        return str(report_file)

# Example usage function
def create_example_statistical_analysis():
    """
    Example of how to use the Statistical Analysis Tools.
    """
    print("📊 Statistical Analysis Tools Example")
    print("=" * 42)
    
    # Create analyzer
    analyzer = StatisticalAnalyzer("prompt_lab_analysis")
    
    # Load data from ledger
    df = analyzer.load_ledger_data()
    
    if len(df) > 0:
        print("\\n🔍 Running comprehensive analysis...")
        
        # Run all analyses
        desc_stats = analyzer.descriptive_statistics(group_by='phase')
        efficiency = analyzer.cost_efficiency_analysis()
        correlations = analyzer.correlation_analysis()
        
        if len(df['phase'].unique()) > 1:
            hypothesis = analyzer.hypothesis_testing('cost_usd', 'phase')
        
        if len(df) > 5:
            trend = analyzer.trend_analysis('cost_usd')
        
        # Generate report
        report_path = analyzer.generate_analysis_report()
        print(f"\\n📝 Full report: {report_path}")
    
    return analyzer

if __name__ == "__main__":
    # Install required packages if not available
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("📦 Installing required packages...")
        import subprocess
        subprocess.check_call(["pip", "install", "matplotlib", "seaborn"])
    
    # Run example
    analyzer = create_example_statistical_analysis()
