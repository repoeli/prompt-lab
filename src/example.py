# example.py

import csv
import datetime

class TokenLedger:
    def __init__(self, filename='data/token_ledger.csv'):
        self.filename = filename
        self.ledger = self.load_ledger()

    def load_ledger(self):
        ledger = []
        try:
            with open(self.filename, mode='r') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    ledger.append(row)
        except FileNotFoundError:
            print(f"{self.filename} not found. Starting with an empty ledger.")
        return ledger

    def add_entry(self, phase, model, tokens_in, tokens_out, cost_usd):
        entry = {
            'date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'phase': phase,
            'model': model,
            'tokens_in': tokens_in,
            'tokens_out': tokens_out,
            'cost_usd': cost_usd
        }
        self.ledger.append(entry)
        self.save_ledger()

    def save_ledger(self):
        with open(self.filename, mode='w', newline='') as file:
            fieldnames = ['date', 'phase', 'model', 'tokens_in', 'tokens_out', 'cost_usd']
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.ledger)

    def get_ledger(self):
        return self.ledger

def main():
    ledger = TokenLedger()
    # Example usage
    ledger.add_entry('Phase 1', 'AI Model', 100, 80, 0.01)
    print(ledger.get_ledger())

if __name__ == "__main__":
    main()