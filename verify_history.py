"""Controlla il report prodotto dalla CLI contro il riferimento versionato."""
import json
from pathlib import Path
import sys

from bankroll import load_matches, market_probabilities


def summarize(report, matches):
    by_match = {(str(m.day), m.home, m.away): m for m in matches}
    model_scores, market_scores = [], []
    for row in report['ledger']:
        match = by_match[row['date'], row['home'], row['away']]
        if match.odds is None:
            continue
        outcome = 0 if match.hg > match.ag else 1 if match.hg == match.ag else 2
        for probabilities, scores in ((row['probabilities'], model_scores),
                                      (market_probabilities(match.odds), market_scores)):
            scores.append(sum((p - (i == outcome)) ** 2 for i, p in enumerate(probabilities)))
    keys = ('mode', 'model', 'source_sha256', 'parameters', 'input_matches', 'predictions',
            'virtual_bets', 'initial_virtual_balance', 'final_virtual_balance', 'virtual_roi',
            'max_daily_drawdown', 'brier', 'log_loss')
    result = {k: report[k] for k in keys}
    result.update(paired_matches=len(model_scores), paired_model_brier=sum(model_scores)/len(model_scores),
                  paired_market_brier=sum(market_scores)/len(market_scores))
    return {k: round(v, 9) if isinstance(v, float) else v for k, v in result.items()}


if __name__ == '__main__':
    report = json.loads(Path(sys.argv[1]).read_text())
    actual = summarize(report, load_matches('storico_serie_a.csv'))
    expected = json.loads(Path('reports/reference.json').read_text())
    if actual != expected:
        raise SystemExit(f'Risultato diverso dal riferimento: {actual}')
    print('Riferimento verificato.')
