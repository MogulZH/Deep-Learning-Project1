"""Print local validation separately from author-confirmed external Kaggle test scores."""
import json
from pathlib import Path

import pandas as pd

root = Path(__file__).resolve().parent
table = pd.read_csv(root/'results/experiment_summary.csv')
scores = json.loads((root/'results/kaggle_scores.json').read_text())
print(table.to_string(index=False))
print('\nTrainable parameters: 2,797,610 (limit: 5,000,000)')
best = scores['best_visible_public']
latest = scores['latest']
print(f'Best reported Kaggle test accuracy: public {best["public_score"]:.2%}; private {best["private_score"]:.2%}')
print('Best recipe: RandAugment, 200 epochs, horizontal-flip probability TTA.')
print(f'Latest reported Kaggle test accuracy: public {latest["public_score"]:.2%}; private {latest["private_score"]:.2%}')
print('Latest recipe: RandAugment + Mixup, 200 epochs, horizontal-flip probability TTA.')
print('These are author-confirmed screenshot scores, not locally computed test accuracy.')
print('All visible screenshot submissions are marked after deadline; no rank/eligibility claim is made.')
