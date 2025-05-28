# 🚀 Professional Prompt Lab

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![OpenAI API](https://img.shields.io/badge/OpenAI-API%20v1.82.0-green.svg)](https://platform.openai.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A professional-grade prompt engineering laboratory for systematic AI experimentation with enterprise-level monitoring, evaluation, and cost tracking.

## 🏗️ Architecture Overview

```
prompt-lab/
├── 📊 data/                    # Data storage and logs
│   ├── token_ledger.csv       # Comprehensive usage tracking
│   ├── evaluations/           # Evaluation results
│   └── exports/               # Data exports
├── 📓 notebooks/              # Jupyter experiments
│   └── 01_foundations.ipynb   # Phase 1 foundations
├── 🔧 src/                    # Core modules
│   ├── runner.py             # Enhanced API interface
│   └── example.py            # Token ledger system
├── 📈 evals/                  # Evaluation framework
│   ├── evaluation_framework.py
│   └── __init__.py
├── 🧪 tests/                  # Test suite
│   └── test_example.py
├── 🔐 .env.example           # Environment template
├── 📦 requirements-pinned.txt # Pinned dependencies
└── 🧪 smoke_test.py          # API validation
```

## 🚀 Quick Start

### 1. **Automated Setup (Recommended)**
```cmd
# Clone and setup in one command
git clone <repository-url>
cd prompt-lab
setup.bat
```

### 2. **Manual Setup**
```cmd
# Create virtual environment
python -m venv .venv
.venv\Scripts\activate.bat

# Install dependencies  
pip install -r requirements-pinned.txt

# Configure environment
copy .env.example .env
# Edit .env and add your OPENAI_API_KEY

# Validate setup
python smoke_test.py
```

## 🔐 Security Setup

### API Key Management
1. **Create `.env` file**:
   ```env
   OPENAI_API_KEY=sk-your-actual-key-here
   PROJECT_NAME=prompt-lab
   LOG_LEVEL=INFO
   ```

2. **Validate configuration**:
   ```cmd
   python smoke_test.py
   ```

3. **Security checklist**:
   - ✅ `.env` in `.gitignore`
   - ✅ No hardcoded keys in code
   - ✅ Smoke test validates key format
   - ✅ Professional error handling

## 📊 Usage Examples

### Basic Usage
```python
from src.runner import run
from src.example import TokenLedger

# Simple API call
response = run("gpt-4o-mini", [
    {"role": "system", "content": "You are an AI writing coach."},
    {"role": "user", "content": "Give me three vivid metaphors for happiness."}
])
```

### Enhanced Usage with Logging
```python
from src.runner import run_with_ledger

# Automatic token tracking and cost logging
response, metrics = run_with_ledger(
    model="gpt-4o-mini",
    messages=messages,
    phase="experiment_1",
    ledger_file="data/token_ledger.csv"
)
```

### Professional Evaluation
```python
from evals.evaluation_framework import PromptEvaluator

evaluator = PromptEvaluator("my_experiment")
result = evaluator.evaluate_response(
    prompt="Your prompt here",
    response=response,
    criteria={"min_length": 50, "contains_keywords": ["specific", "terms"]}
)
```

## 🎯 Features

### 🔧 **Core Capabilities**
- ✅ Modern OpenAI API (v1.82.0) with Chat Completions
- ✅ Comprehensive token usage tracking  
- ✅ Real-time cost monitoring and analysis
- ✅ Professional error handling and logging
- ✅ Secure API key management

### 📊 **Analytics & Monitoring**  
- ✅ Token ledger with CSV persistence
- ✅ Cost efficiency metrics
- ✅ Performance benchmarking (latency, throughput)
- ✅ Quality evaluation framework
- ✅ Automated report generation

### 🧪 **Evaluation Framework**
- ✅ Multi-criteria response evaluation
- ✅ Keyword coverage analysis
- ✅ Quality metrics and scoring
- ✅ Batch evaluation capabilities
- ✅ Professional reporting (JSON/CSV)

### 🛡️ **Enterprise Ready**
- ✅ Pinned dependencies for reproducibility
- ✅ Professional directory structure  
- ✅ Comprehensive error handling
- ✅ Production-grade logging
- ✅ Security best practices

## 📈 Token Ledger System

The professional token ledger tracks all API usage:

| Field | Description | Example |
|-------|-------------|---------|
| `date` | Timestamp | `2025-05-28 10:30:15` |
| `phase` | Experiment phase | `phase1`, `evaluation` |
| `model` | Model used | `gpt-4o-mini` |
| `tokens_in` | Input tokens | `42` |
| `tokens_out` | Output tokens | `156` |
| `cost_usd` | Cost in USD | `0.000423` |

### Cost Analysis Commands
```python
# Load and analyze ledger
ledger = TokenLedger("data/token_ledger.csv")
entries = ledger.get_ledger()

# Calculate total costs
total_cost = sum(float(e['cost_usd']) for e in entries)
phase1_cost = sum(float(e['cost_usd']) for e in entries if e['phase'] == 'phase1')
```

## 🧪 Testing & Validation

### Smoke Test
```cmd
python smoke_test.py
```
**Validates**:
- ✅ API key format and authentication
- ✅ OpenAI client initialization  
- ✅ Basic API functionality
- ✅ Response parsing and metrics
- ✅ Cost calculation accuracy

### Professional Test Suite
```cmd
# Run all tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=html
```

## 📚 Jupyter Notebooks

### 01_foundations.ipynb
**Phase 1 Complete Foundation**:
- ✅ Modern API integration demo
- ✅ Token ledger system showcase  
- ✅ Multi-scenario testing
- ✅ Professional evaluation framework
- ✅ Cost analysis and reporting
- ✅ Quality metrics and benchmarks

**Start experimenting**:
```cmd
jupyter lab notebooks/01_foundations.ipynb
```

## 💰 Cost Management

### Pricing (as of May 2025)
| Model | Input (per 1M tokens) | Output (per 1M tokens) |
|-------|----------------------|------------------------|
| `gpt-4o-mini` | $0.60 | $2.40 |

### Cost Monitoring Features
- 🔍 **Real-time tracking**: Every API call logged
- 📊 **Analytics**: Cost per phase, session, token
- ⚡ **Efficiency**: Output tokens per dollar metrics  
- 📈 **Reporting**: Automated cost analysis
- 🎯 **Budgeting**: Track and optimize spending

## 🔧 Advanced Configuration

### Environment Variables
```env
# Required
OPENAI_API_KEY=sk-your-key-here

# Optional
OPENAI_ORG_ID=org-your-org-id
PROJECT_NAME=prompt-lab
LOG_LEVEL=INFO
MAX_TOKENS=1000
TEMPERATURE=0.7
```

### Custom Pricing
```python
# Update pricing in runner.py
PRICE = {
    "gpt-4o-mini": {"in": 0.60e-6, "out": 2.40e-6},
    "gpt-4": {"in": 30e-6, "out": 60e-6},
    # Add custom models
}
```

## 🚀 Next Steps: Phase 2

Phase 1 establishes the foundation. Phase 2 will include:
- 🔬 **Advanced prompt engineering techniques**
- 🎯 **A/B testing framework**  
- 📊 **Statistical analysis tools**
- 🤖 **Multi-model comparison**
- 🔄 **Automated optimization**
- 📈 **Advanced analytics dashboard**

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- 📧 **Issues**: Use GitHub Issues for bug reports
- 📚 **Documentation**: Check the notebooks for examples  
- 🧪 **Setup Problems**: Run `python smoke_test.py` for diagnostics
- 💬 **Questions**: Open a GitHub Discussion

---

**🎉 Ready to start professional prompt engineering experiments!**