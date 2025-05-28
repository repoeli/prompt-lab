"""
Advanced Analytics Dashboard

This module provides comprehensive analytics and visualization capabilities
for prompt engineering experiments, including real-time monitoring,
interactive dashboards, and performance tracking.

Key Features:
- Interactive visualizations with Plotly
- Real-time performance monitoring
- Cost tracking and efficiency dashboards
- Trend analysis and alerts
- Comparative analytics across experiments
- Export capabilities for reports
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from datetime import datetime, timedelta
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import seaborn as sns
import matplotlib.pyplot as plt
from io import StringIO
import base64

@dataclass
class DashboardConfig:
    """Configuration for dashboard appearance and behavior."""
    theme: str = "plotly_white"
    color_palette: List[str] = None
    show_grid: bool = True
    show_legend: bool = True
    height: int = 600
    width: int = 800
    update_interval: int = 30  # seconds
    
    def __post_init__(self):
        if self.color_palette is None:
            self.color_palette = [
                '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
            ]

class PerformanceMonitor:
    """
    Real-time performance monitoring for prompt experiments.
    
    Tracks key metrics and provides alerts when performance deviates
    from expected ranges.
    """
    
    def __init__(self, config: DashboardConfig = None):
        self.config = config or DashboardConfig()
        self.alerts: List[Dict] = []
        self.thresholds = {
            'response_time_max': 10.0,  # seconds
            'cost_per_request_max': 0.05,  # dollars
            'error_rate_max': 0.05,  # 5%
            'token_efficiency_min': 0.7  # tokens used / tokens available
        }
        
    def check_performance_alerts(self, data: pd.DataFrame) -> List[Dict]:
        """Check for performance issues and generate alerts."""
        alerts = []
        current_time = datetime.now()
        
        if data.empty:
            return alerts
        
        # Check response time
        if 'response_time' in data.columns:
            avg_response_time = data['response_time'].mean()
            if avg_response_time > self.thresholds['response_time_max']:
                alerts.append({
                    'type': 'performance',
                    'severity': 'warning',
                    'message': f'High response time: {avg_response_time:.2f}s (threshold: {self.thresholds["response_time_max"]}s)',
                    'timestamp': current_time,
                    'metric': 'response_time',
                    'value': avg_response_time
                })
        
        # Check cost efficiency
        if 'total_cost' in data.columns:
            avg_cost = data['total_cost'].mean()
            if avg_cost > self.thresholds['cost_per_request_max']:
                alerts.append({
                    'type': 'cost',
                    'severity': 'warning',
                    'message': f'High cost per request: ${avg_cost:.4f} (threshold: ${self.thresholds["cost_per_request_max"]})',
                    'timestamp': current_time,
                    'metric': 'cost_per_request',
                    'value': avg_cost
                })
        
        # Check error rate
        if 'error' in data.columns:
            error_rate = data['error'].sum() / len(data)
            if error_rate > self.thresholds['error_rate_max']:
                alerts.append({
                    'type': 'reliability',
                    'severity': 'critical',
                    'message': f'High error rate: {error_rate:.1%} (threshold: {self.thresholds["error_rate_max"]:.1%})',
                    'timestamp': current_time,
                    'metric': 'error_rate',
                    'value': error_rate
                })
        
        self.alerts.extend(alerts)
        return alerts

class VisualizationEngine:
    """
    Core visualization engine using Plotly for interactive charts.
    
    Provides various chart types optimized for prompt engineering analytics.
    """
    
    def __init__(self, config: DashboardConfig = None):
        self.config = config or DashboardConfig()
        
    def create_performance_timeline(self, data: pd.DataFrame) -> go.Figure:
        """Create a timeline showing performance metrics over time."""
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=('Response Time', 'Cost per Request', 'Token Usage'),
            vertical_spacing=0.1
        )
        
        if 'timestamp' not in data.columns:
            data['timestamp'] = pd.date_range(start='2024-01-01', periods=len(data), freq='h')
        
        # Response time
        if 'response_time' in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data['timestamp'],
                    y=data['response_time'],
                    mode='lines+markers',
                    name='Response Time',
                    line=dict(color=self.config.color_palette[0])
                ),
                row=1, col=1
            )
        
        # Cost
        if 'total_cost' in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data['timestamp'],
                    y=data['total_cost'],
                    mode='lines+markers',
                    name='Cost',
                    line=dict(color=self.config.color_palette[1])
                ),
                row=2, col=1
            )
        
        # Token usage
        if 'total_tokens' in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data['timestamp'],
                    y=data['total_tokens'],
                    mode='lines+markers',
                    name='Tokens',
                    line=dict(color=self.config.color_palette[2])
                ),
                row=3, col=1
            )
        
        fig.update_layout(
            height=self.config.height,
            title_text="📈 Performance Timeline",
            showlegend=False
        )
        
        return fig
    
    def create_cost_analysis_dashboard(self, data: pd.DataFrame) -> go.Figure:
        """Create comprehensive cost analysis visualizations."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Cost Distribution', 'Cost vs Performance',
                'Daily Cost Trend', 'Model Cost Comparison'
            ),
            specs=[[{"type": "histogram"}, {"type": "scatter"}],
                   [{"type": "scatter"}, {"type": "bar"}]]
        )
        
        if 'total_cost' not in data.columns:
            return fig
        
        # Cost distribution
        fig.add_trace(
            go.Histogram(
                x=data['total_cost'],
                name='Cost Distribution',
                marker_color=self.config.color_palette[0]
            ),
            row=1, col=1
        )
        
        # Cost vs Performance scatter
        if 'response_time' in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data['total_cost'],
                    y=data['response_time'],
                    mode='markers',
                    name='Cost vs Response Time',
                    marker=dict(
                        color=self.config.color_palette[1],
                        size=8,
                        opacity=0.6
                    )
                ),
                row=1, col=2
            )
        
        # Daily cost trend
        if 'timestamp' in data.columns:
            daily_costs = data.groupby(data['timestamp'].dt.date)['total_cost'].sum().reset_index()
            fig.add_trace(
                go.Scatter(
                    x=daily_costs['timestamp'],
                    y=daily_costs['total_cost'],
                    mode='lines+markers',
                    name='Daily Cost',
                    line=dict(color=self.config.color_palette[2])
                ),
                row=2, col=1
            )
        
        # Model cost comparison
        if 'model' in data.columns:
            model_costs = data.groupby('model')['total_cost'].mean().reset_index()
            fig.add_trace(
                go.Bar(
                    x=model_costs['model'],
                    y=model_costs['total_cost'],
                    name='Avg Cost by Model',
                    marker_color=self.config.color_palette[3]
                ),
                row=2, col=2
            )
        
        fig.update_layout(
            height=self.config.height * 1.2,
            title_text="💰 Cost Analysis Dashboard",
            showlegend=False
        )
        
        return fig
    
    def create_ab_test_comparison(self, ab_results: Dict) -> go.Figure:
        """Create visualizations for A/B test results."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Performance Comparison', 'Statistical Significance',
                'Effect Size Analysis', 'Confidence Intervals'
            )
        )
        
        if 'variants' not in ab_results:
            return fig
        
        variants = ab_results['variants']
        variant_names = list(variants.keys())
        
        # Performance comparison
        performance_scores = [variants[name].get('mean_score', 0) for name in variant_names]
        fig.add_trace(
            go.Bar(
                x=variant_names,
                y=performance_scores,
                name='Performance',
                marker_color=self.config.color_palette[:len(variant_names)]
            ),
            row=1, col=1
        )
        
        # Statistical significance (p-values)
        if 'statistical_results' in ab_results:
            p_values = [ab_results['statistical_results'].get(f'{name}_pvalue', 1.0) for name in variant_names]
            fig.add_trace(
                go.Bar(
                    x=variant_names,
                    y=p_values,
                    name='P-values',
                    marker_color='red'
                ),
                row=1, col=2
            )
            # Add significance line
            fig.add_hline(y=0.05, line_dash="dash", line_color="red", row=1, col=2)
        
        # Effect sizes
        if 'effect_sizes' in ab_results:
            effect_sizes = [ab_results['effect_sizes'].get(name, 0) for name in variant_names]
            fig.add_trace(
                go.Bar(
                    x=variant_names,
                    y=effect_sizes,
                    name='Effect Size',
                    marker_color=self.config.color_palette[2]
                ),
                row=2, col=1
            )
        
        # Confidence intervals
        if 'confidence_intervals' in ab_results:
            for i, name in enumerate(variant_names):
                ci = ab_results['confidence_intervals'].get(name, [0, 0])
                fig.add_trace(
                    go.Scatter(
                        x=[name, name],
                        y=ci,
                        mode='lines+markers',
                        name=f'{name} CI',
                        line=dict(color=self.config.color_palette[i], width=3)
                    ),
                    row=2, col=2
                )
        
        fig.update_layout(
            height=self.config.height * 1.2,
            title_text="🧪 A/B Test Results Dashboard",
            showlegend=False
        )
        
        return fig
    
    def create_model_comparison_radar(self, comparison_data: Dict) -> go.Figure:
        """Create radar chart for model comparison."""
        if 'models' not in comparison_data:
            return go.Figure()
        
        models = comparison_data['models']
        metrics = ['performance', 'cost_efficiency', 'speed', 'reliability', 'token_efficiency']
        
        fig = go.Figure()
        
        for model_name, model_data in models.items():
            # Normalize metrics to 0-1 scale for radar chart
            values = []
            for metric in metrics:
                value = model_data.get(metric, 0.5)
                # Normalize to 0-1 scale
                if metric == 'cost_efficiency':
                    values.append(min(1.0, value))
                elif metric == 'speed':
                    values.append(min(1.0, 1.0 / max(0.1, value)))  # Invert speed (lower is better)
                else:
                    values.append(min(1.0, value))
            
            values.append(values[0])  # Close the radar chart
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=metrics + [metrics[0]],
                fill='toself',
                name=model_name,
                opacity=0.6
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            title="🤖 Model Performance Comparison",
            height=self.config.height
        )
        
        return fig
    
    def create_optimization_progress(self, optimization_history: List[Dict]) -> go.Figure:
        """Create visualization for optimization progress."""
        if not optimization_history:
            return go.Figure()
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Fitness Over Generations', 'Population Diversity')
        )
        
        for i, run in enumerate(optimization_history):
            if 'generations_history' in run:
                generations = run['generations_history']
                gen_numbers = [g['generation'] for g in generations]
                best_fitness = [g['best_fitness'] for g in generations]
                avg_fitness = [g['avg_fitness'] for g in generations]
                
                # Best fitness
                fig.add_trace(
                    go.Scatter(
                        x=gen_numbers,
                        y=best_fitness,
                        mode='lines+markers',
                        name=f'Run {i+1} - Best',
                        line=dict(color=self.config.color_palette[i % len(self.config.color_palette)])
                    ),
                    row=1, col=1
                )
                
                # Average fitness
                fig.add_trace(
                    go.Scatter(
                        x=gen_numbers,
                        y=avg_fitness,
                        mode='lines',
                        name=f'Run {i+1} - Avg',
                        line=dict(
                            color=self.config.color_palette[i % len(self.config.color_palette)],
                            dash='dash'
                        )
                    ),
                    row=1, col=1
                )
                
                # Diversity (difference between best and avg)
                diversity = [b - a for b, a in zip(best_fitness, avg_fitness)]
                fig.add_trace(
                    go.Scatter(
                        x=gen_numbers,
                        y=diversity,
                        mode='lines',
                        name=f'Run {i+1} - Diversity',
                        line=dict(color=self.config.color_palette[i % len(self.config.color_palette)])
                    ),
                    row=2, col=1
                )
        
        fig.update_layout(
            height=self.config.height,
            title_text="🧬 Optimization Progress Dashboard"
        )
        
        return fig

class DashboardGenerator:
    """
    Main dashboard generator that combines all visualization components.
    
    Creates comprehensive analytics dashboards for prompt engineering experiments.
    """
    
    def __init__(self, config: DashboardConfig = None):
        self.config = config or DashboardConfig()
        self.viz_engine = VisualizationEngine(self.config)
        self.monitor = PerformanceMonitor(self.config)
        
    def load_token_ledger_data(self, filepath: str) -> pd.DataFrame:
        """Load and preprocess token ledger data."""
        try:
            data = pd.read_csv(filepath)
            
            # Convert timestamp if it exists
            if 'timestamp' in data.columns:
                data['timestamp'] = pd.to_datetime(data['timestamp'])
            
            # Calculate derived metrics
            if 'total_cost' not in data.columns and 'input_cost' in data.columns and 'output_cost' in data.columns:
                data['total_cost'] = data['input_cost'] + data['output_cost']
            
            if 'total_tokens' not in data.columns and 'input_tokens' in data.columns and 'output_tokens' in data.columns:
                data['total_tokens'] = data['input_tokens'] + data['output_tokens']
            
            # Add efficiency metrics
            if 'response_time' in data.columns and 'total_tokens' in data.columns:
                data['tokens_per_second'] = data['total_tokens'] / data['response_time'].clip(lower=0.1)
            
            if 'total_cost' in data.columns and 'total_tokens' in data.columns:
                data['cost_per_token'] = data['total_cost'] / data['total_tokens'].clip(lower=1)
            
            return data
            
        except FileNotFoundError:
            print(f"Warning: Token ledger file not found at {filepath}")
            return pd.DataFrame()
        except Exception as e:
            print(f"Error loading token ledger: {e}")
            return pd.DataFrame()
    
    def generate_executive_dashboard(self, data_sources: Dict[str, Any]) -> Dict[str, go.Figure]:
        """Generate a comprehensive executive dashboard."""
        figures = {}
        
        # Load main data
        if 'token_ledger_path' in data_sources:
            main_data = self.load_token_ledger_data(data_sources['token_ledger_path'])
        else:
            main_data = data_sources.get('main_data', pd.DataFrame())
        
        # Performance timeline
        if not main_data.empty:
            figures['performance_timeline'] = self.viz_engine.create_performance_timeline(main_data)
            figures['cost_analysis'] = self.viz_engine.create_cost_analysis_dashboard(main_data)
        
        # A/B test results
        if 'ab_results' in data_sources:
            figures['ab_test_comparison'] = self.viz_engine.create_ab_test_comparison(data_sources['ab_results'])
        
        # Model comparison
        if 'model_comparison' in data_sources:
            figures['model_comparison'] = self.viz_engine.create_model_comparison_radar(data_sources['model_comparison'])
        
        # Optimization progress
        if 'optimization_history' in data_sources:
            figures['optimization_progress'] = self.viz_engine.create_optimization_progress(data_sources['optimization_history'])
        
        return figures
    
    def create_real_time_monitor(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Create real-time monitoring dashboard with alerts."""
        monitor_data = {
            'alerts': self.monitor.check_performance_alerts(data),
            'current_metrics': {},
            'trend_indicators': {}
        }
        
        if not data.empty:
            # Current metrics
            monitor_data['current_metrics'] = {
                'avg_response_time': data['response_time'].mean() if 'response_time' in data.columns else 0,
                'total_cost_today': data['total_cost'].sum() if 'total_cost' in data.columns else 0,
                'requests_today': len(data),
                'error_rate': (data['error'].sum() / len(data)) if 'error' in data.columns else 0,
                'avg_tokens_per_request': data['total_tokens'].mean() if 'total_tokens' in data.columns else 0
            }
            
            # Trend indicators (comparing with previous period)
            if len(data) > 10:
                midpoint = len(data) // 2
                recent_data = data.iloc[midpoint:]
                older_data = data.iloc[:midpoint]
                
                monitor_data['trend_indicators'] = {
                    'response_time_trend': self._calculate_trend(
                        older_data['response_time'].mean() if 'response_time' in older_data.columns else 0,
                        recent_data['response_time'].mean() if 'response_time' in recent_data.columns else 0
                    ),
                    'cost_trend': self._calculate_trend(
                        older_data['total_cost'].mean() if 'total_cost' in older_data.columns else 0,
                        recent_data['total_cost'].mean() if 'total_cost' in recent_data.columns else 0
                    )
                }
        
        return monitor_data
    
    def _calculate_trend(self, old_value: float, new_value: float) -> Dict[str, Any]:
        """Calculate trend direction and percentage change."""
        if old_value == 0:
            return {'direction': 'stable', 'change_percent': 0}
        
        change_percent = ((new_value - old_value) / old_value) * 100
        
        if abs(change_percent) < 5:
            direction = 'stable'
        elif change_percent > 0:
            direction = 'increasing'
        else:
            direction = 'decreasing'
        
        return {
            'direction': direction,
            'change_percent': round(change_percent, 1)
        }
    
    def export_dashboard_html(self, figures: Dict[str, go.Figure], output_path: str) -> None:
        """Export dashboard as standalone HTML file."""
        html_content = [
            "<!DOCTYPE html>",
            "<html>",
            "<head>",
            "<title>Prompt Lab Analytics Dashboard</title>",
            "<script src='https://cdn.plot.ly/plotly-latest.min.js'></script>",
            "<style>",
            "body { font-family: Arial, sans-serif; margin: 20px; }",
            ".dashboard-section { margin-bottom: 30px; }",
            ".chart-container { margin: 20px 0; }",
            "h1 { color: #2c3e50; text-align: center; }",
            "h2 { color: #34495e; border-bottom: 2px solid #3498db; padding-bottom: 10px; }",
            "</style>",
            "</head>",
            "<body>",
            "<h1>📊 Prompt Lab Analytics Dashboard</h1>",
            f"<p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>"
        ]
        
        for title, fig in figures.items():
            # Convert figure to HTML
            fig_html = fig.to_html(include_plotlyjs=False, div_id=title)
            
            html_content.extend([
                f"<div class='dashboard-section'>",
                f"<h2>{title.replace('_', ' ').title()}</h2>",
                f"<div class='chart-container'>{fig_html}</div>",
                "</div>"
            ])
        
        html_content.extend([
            "</body>",
            "</html>"
        ])
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(html_content))
        
        print(f"📊 Dashboard exported to {output_path}")
    
    def generate_summary_report(self, data_sources: Dict[str, Any]) -> str:
        """Generate a text summary of key insights."""
        report_lines = [
            "# 📈 Prompt Lab Analytics Summary",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            ""
        ]
        
        # Load main data for analysis
        if 'token_ledger_path' in data_sources:
            data = self.load_token_ledger_data(data_sources['token_ledger_path'])
        else:
            data = data_sources.get('main_data', pd.DataFrame())
        
        if not data.empty:
            report_lines.extend([
                "## Key Metrics",
                f"- Total requests: {len(data):,}",
                f"- Average response time: {data['response_time'].mean():.2f}s" if 'response_time' in data.columns else "",
                f"- Total cost: ${data['total_cost'].sum():.4f}" if 'total_cost' in data.columns else "",
                f"- Average cost per request: ${data['total_cost'].mean():.4f}" if 'total_cost' in data.columns else "",
                f"- Total tokens used: {data['total_tokens'].sum():,}" if 'total_tokens' in data.columns else "",
                ""
            ])
            
            # Performance insights
            if 'response_time' in data.columns:
                slow_requests = (data['response_time'] > 5).sum()
                report_lines.extend([
                    "## Performance Insights",
                    f"- Requests > 5s: {slow_requests} ({slow_requests/len(data)*100:.1f}%)",
                    f"- Fastest response: {data['response_time'].min():.2f}s",
                    f"- Slowest response: {data['response_time'].max():.2f}s",
                    ""
                ])
            
            # Cost insights
            if 'total_cost' in data.columns:
                expensive_requests = (data['total_cost'] > 0.01).sum()
                report_lines.extend([
                    "## Cost Insights",
                    f"- Requests > $0.01: {expensive_requests} ({expensive_requests/len(data)*100:.1f}%)",
                    f"- Most expensive request: ${data['total_cost'].max():.4f}",
                    f"- Cost efficiency: {data['total_tokens'].sum() / data['total_cost'].sum():.0f} tokens/$" if data['total_cost'].sum() > 0 else "",
                    ""
                ])
        
        # Add alerts if any
        alerts = self.monitor.check_performance_alerts(data)
        if alerts:
            report_lines.extend([
                "## 🚨 Alerts",
                ""
            ])
            for alert in alerts[-5:]:  # Show last 5 alerts
                report_lines.append(f"- **{alert['severity'].upper()}**: {alert['message']}")
            report_lines.append("")
        
        # Recommendations
        report_lines.extend([
            "## 💡 Recommendations",
            ""
        ])
        
        if not data.empty:
            if 'response_time' in data.columns and data['response_time'].mean() > 3:
                report_lines.append("- Consider optimizing prompts to reduce response time")
            if 'total_cost' in data.columns and data['total_cost'].mean() > 0.02:
                report_lines.append("- Review cost efficiency - consider shorter prompts or different models")
            if len(data) < 50:
                report_lines.append("- Increase sample size for more reliable analytics")
        
        report_lines.extend([
            "- Regular monitoring recommended for optimal performance",
            "- Consider A/B testing for prompt optimization",
            "- Set up automated alerts for cost and performance thresholds"
        ])
        
        return "\n".join(report_lines)


# Example usage and testing functions
def create_sample_dashboard_data():
    """Create sample data for dashboard testing."""
    import random
    from datetime import datetime, timedelta
    
    # Generate sample token ledger data
    n_samples = 100
    start_date = datetime.now() - timedelta(days=7)
    
    data = []
    for i in range(n_samples):
        timestamp = start_date + timedelta(hours=i * 2)
        data.append({
            'timestamp': timestamp,
            'model': random.choice(['gpt-4o-mini', 'gpt-3.5-turbo']),
            'input_tokens': random.randint(50, 500),
            'output_tokens': random.randint(20, 200),
            'input_cost': random.uniform(0.001, 0.01),
            'output_cost': random.uniform(0.002, 0.02),
            'response_time': random.uniform(0.5, 8.0),
            'error': random.choice([0, 0, 0, 0, 1])  # 20% error rate
        })
    
    df = pd.DataFrame(data)
    df['total_cost'] = df['input_cost'] + df['output_cost']
    df['total_tokens'] = df['input_tokens'] + df['output_tokens']
    
    return df

if __name__ == "__main__":
    # Example usage
    print("🚀 Testing Advanced Analytics Dashboard")
    
    # Create sample data
    sample_data = create_sample_dashboard_data()
    
    # Initialize dashboard
    dashboard = DashboardGenerator()
    
    # Create data sources
    data_sources = {
        'main_data': sample_data,
        'ab_results': {
            'variants': {
                'Control': {'mean_score': 0.75, 'std_score': 0.1},
                'Variant A': {'mean_score': 0.82, 'std_score': 0.12},
                'Variant B': {'mean_score': 0.78, 'std_score': 0.09}
            },
            'statistical_results': {
                'Control_pvalue': 1.0,
                'Variant A_pvalue': 0.02,
                'Variant B_pvalue': 0.15
            }
        }
    }
    
    # Generate dashboard
    figures = dashboard.generate_executive_dashboard(data_sources)
    
    print(f"✅ Generated {len(figures)} dashboard charts")
    
    # Generate summary report
    summary = dashboard.generate_summary_report(data_sources)
    print("\n📊 Summary Report Preview:")
    print(summary[:500] + "..." if len(summary) > 500 else summary)
