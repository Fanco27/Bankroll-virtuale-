"""Baseline Poisson, validazione cronologica e contabilita esclusivamente virtuale."""
import argparse
import csv
from dataclasses import dataclass
from datetime import date, datetime
import hashlib
import json
import math
from pathlib import Path


@dataclass(frozen=True)
class Match:
    day: date
    home: str
    away: str
    hg: int
    ag: int
    odds: tuple | None = None


def load_matches(path):
    """Football-Data CSV. Quote B365: il momento di rilevazione non e verificato."""
    matches, seen = [], set()
    with open(path, encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        required = {'Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG'}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError('Colonne richieste: ' + ', '.join(sorted(required)))
        for row in reader:
            if not any(row.values()):
                continue
            raw = row['Date'].strip()
            day = None
            for fmt in ('%d/%m/%Y', '%d/%m/%y', '%Y-%m-%d'):
                try:
                    day = datetime.strptime(raw, fmt).date()
                    break
                except ValueError:
                    pass
            if day is None:
                raise ValueError('Data non valida: ' + raw)
            home, away = row['HomeTeam'].strip(), row['AwayTeam'].strip()
            hg, ag = int(row['FTHG']), int(row['FTAG'])
            if not home or not away or home == away or min(hg, ag) < 0:
                raise ValueError('Partita non valida')
            key = (day, home, away)
            if key in seen:
                raise ValueError('Partita duplicata: ' + str(key))
            seen.add(key)
            values = [row.get(k, '').strip() for k in ('B365H', 'B365D', 'B365A')]
            if any(values) and not all(values):
                raise ValueError('Quote parziali: ' + str(key))
            odds = tuple(map(float, values)) if all(values) else None
            if odds is not None:
                market_probabilities(odds)
            matches.append(Match(day, home, away, hg, ag, odds))
    return sorted(matches, key=lambda m: (m.day, m.home, m.away))


def market_probabilities(odds):
    if len(odds) != 3 or any(not math.isfinite(o) or o <= 1 for o in odds):
        raise ValueError('Servono tre quote decimali finite maggiori di 1')
    inverse = [1 / o for o in odds]
    total = sum(inverse)
    return tuple(p / total for p in inverse)


def poisson_probs(home_rate, away_rate):
    """1/X/2 da Poisson indipendenti; coda troncata < 1e-12 per squadra."""
    def pmf(rate):
        if not math.isfinite(rate) or not 0 < rate <= 20:
            raise ValueError('Intensita Poisson fuori intervallo (0, 20]')
        values = [math.exp(-rate)]
        while 1 - sum(values) > 1e-12:
            values.append(values[-1] * rate / len(values))
        return values
    hp, ap = pmf(home_rate), pmf(away_rate)
    result = [0., 0., 0.]
    for h, ph in enumerate(hp):
        for a, pa in enumerate(ap):
            result[0 if h > a else 1 if h == a else 2] += ph * pa
    total = sum(result)
    return tuple(p / total for p in result)


class PoissonModel:
    """Medie casa/trasferta con regolarizzazione verso la media del campionato."""
    def __init__(self, prior=5.):
        if not math.isfinite(prior) or prior <= 0:
            raise ValueError('Prior deve essere positivo')
        self.prior = prior
        self.history = []

    def fit(self, history, before):
        self.history = [m for m in history if m.day < before]
        if not self.history:
            raise ValueError('Nessuna partita precedente alla previsione')
        self.last_training_date = max(m.day for m in self.history)
        return self

    def predict(self, home, away):
        if not self.history:
            raise ValueError('Modello non addestrato')
        base_h = max(.05, sum(m.hg for m in self.history) / len(self.history))
        base_a = max(.05, sum(m.ag for m in self.history) / len(self.history))
        hs = [m for m in self.history if m.home == home]
        ats = [m for m in self.history if m.away == away]
        def mean(rows, attr, base):
            return (sum(getattr(m, attr) for m in rows) + self.prior * base) / (len(rows) + self.prior)
        lh = mean(hs, 'hg', base_h) * mean(ats, 'hg', base_h) / base_h
        la = mean(ats, 'ag', base_a) * mean(hs, 'ag', base_a) / base_a
        return poisson_probs(min(20., max(.05, lh)), min(20., max(.05, la)))


def backtest(matches, min_train=100, initial=1000., stake_fraction=.01, min_edge=.05):
    """Un solo esito per match. Puntate giornaliere fissate prima dei risultati."""
    if min_train < 1 or not math.isfinite(initial) or initial <= 0:
        raise ValueError('Training e saldo iniziale devono essere positivi')
    if not math.isfinite(stake_fraction) or not 0 < stake_fraction <= .05:
        raise ValueError('Frazione virtuale consentita: (0, 0.05]')
    if not math.isfinite(min_edge) or min_edge < 0:
        raise ValueError('Soglia valore non valida')
    ordered = sorted(matches, key=lambda m: (m.day, m.home, m.away))
    history, ledger = [], []
    balance, peak, max_dd, turnover = initial, initial, 0., 0.
    brier, logloss, market_brier = [], [], []
    for day in sorted({m.day for m in ordered}):
        batch = [m for m in ordered if m.day == day]
        if len(history) >= min_train:
            model = PoissonModel().fit(history, day)
            pending = []
            for m in batch:
                p = model.predict(m.home, m.away)
                outcome = 0 if m.hg > m.ag else 1 if m.hg == m.ag else 2
                brier.append(sum((p[i] - (i == outcome)) ** 2 for i in range(3)))
                logloss.append(-math.log(max(p[outcome], 1e-15)))
                pick = None
                if m.odds:
                    mp = market_probabilities(m.odds)
                    market_brier.append(sum((mp[i] - (i == outcome)) ** 2 for i in range(3)))
                    edges = [p[i] * m.odds[i] - 1 for i in range(3)]
                    best = max(range(3), key=edges.__getitem__)
                    if edges[best] > min_edge:
                        pick = best
                pending.append((m, p, outcome, pick))
            count = sum(pick is not None for _, _, _, pick in pending)
            stake = min(balance * stake_fraction, balance / count) if count else 0.
            day_profit = 0.
            for m, p, outcome, pick in pending:
                amount = stake if pick is not None else 0.
                profit = amount * (m.odds[pick] - 1) if pick == outcome else -amount
                day_profit += profit
                turnover += amount
                ledger.append({'date': str(day), 'home': m.home, 'away': m.away,
                               'training_end': str(model.last_training_date),
                               'probabilities': p, 'outcome': outcome,
                               'selection': pick, 'virtual_stake': amount,
                               'virtual_profit': profit})
            balance += day_profit
            peak = max(peak, balance)
            max_dd = max(max_dd, (peak - balance) / peak)
        history.extend(batch)
    average = lambda xs: sum(xs) / len(xs) if xs else None
    return {'mode': 'VIRTUAL_ONLY', 'model': 'poisson_baseline_v1',
            'status': 'RESEARCH_NOT_VALIDATED', 'input_matches': len(matches),
            'predictions': len(ledger), 'virtual_bets': sum(r['selection'] is not None for r in ledger),
            'initial_virtual_balance': initial, 'final_virtual_balance': balance,
            'virtual_turnover': turnover, 'virtual_roi': (balance-initial)/turnover if turnover else None,
            'max_daily_drawdown': max_dd, 'brier': average(brier), 'log_loss': average(logloss),
            'market_brier': average(market_brier), 'market_sample_size': len(market_brier),
            'parameters': {'min_train': min_train, 'stake_fraction': stake_fraction, 'min_edge': min_edge},
            'limitations': ['Quote storiche: timestamp e disponibilita pre-partita non verificati.',
                            'Modello baseline, nessuna prova di redditivita futura.',
                            'Il Brier mercato usa solo partite con quote: confrontare campioni uguali.'],
            'ledger': ledger}


def main():
    parser = argparse.ArgumentParser(description='Ricerca esclusivamente virtuale, CSV Football-Data')
    parser.add_argument('csv', type=Path)
    parser.add_argument('--min-train', type=int, default=100)
    parser.add_argument('--output', type=Path, default=Path('report.json'))
    args = parser.parse_args()
    report = backtest(load_matches(args.csv), min_train=args.min_train)
    if not report['predictions']:
        raise SystemExit('Dati insufficienti: nessuna previsione fuori campione')
    report['source_sha256'] = hashlib.sha256(args.csv.read_bytes()).hexdigest()
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(f"VIRTUAL_ONLY: {report['predictions']} previsioni; report {args.output}")


if __name__ == '__main__':
    main()
