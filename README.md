# My Python App

## Overview
This project is a Python application designed to interact with a chat model. It includes functionalities for tracking token usage and provides a framework for testing and evaluation.

## Directory Structure
```
prompt-lab/
├── data/
│   └── token_ledger.csv
├── notebooks/
│   └── phase1.ipynb
├── src/
│   ├── __init__.py
│   └── example.py
├── tests/
│   └── test_example.py
├── .gitignore
├── requirements.txt
└── README.md
```

## Installation
To set up the project, clone the repository and install the required dependencies:

```bash
git clone <repository-url>
cd prompt-lab
pip install -r requirements.txt
```

## Usage
To run the application, execute the following command:

```bash
python src/example.py
```

This will initiate a chat session with the model.

## Token Ledger
The `data/token_ledger.csv` file is used to track token usage. It includes the following columns:
- date
- phase
- model
- tokens_in
- tokens_out
- cost_usd

## Jupyter Notebook
The `notebooks/phase1.ipynb` file contains Python code to interact with the AI writing coach.

## Testing
To run the tests, use the following command:

```bash
pytest tests/
```

## Contribution
Contributions are welcome! Please submit a pull request or open an issue for any enhancements or bug fixes.

## License
This project is licensed under the MIT License. See the LICENSE file for details.