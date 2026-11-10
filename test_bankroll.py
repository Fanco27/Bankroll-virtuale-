import csv
from datetime import date, timedelta
import math
from pathlib import Path
import tempfile
import unittest
from bankroll import Match, PoissonModel, backtest, load_matches, market_probabilities, poisson_probs


def fixture():
    return [Match(date(2024, 1, 1) + timedelta(days=i//2),
                  'A' if i % 2 == 0 else 'C', 'B' if i % 2 == 0 else 'D',
                  i % 4, (i + 1) % 3, (2.2, 3.2, 3.4)) for i in range(40)]


class ModelTests(unittest.TestCase):
    def test_market_normalization(self):
        self.assertEqual(market_probabilities((2., 2., 2.)), (1/3, 1/3, 1/3))

    def test_invalid_odds(self):
        for odds in ((1, 3, 4), (float('nan'), 3, 4), (float('inf'), 3, 4), (2, 3)):
            with self.assertRaises(ValueError):
                market_probabilities(odds)

    def test_poisson_symmetry_and_normalization(self):
        p = poisson_probs(1.4, 1.4)
        self.assertAlmostEqual(sum(p), 1.)
        self.assertAlmostEqual(p[0], p[2])
        self.assertTrue(all(0 < x < 1 for x in p))
        self.assertGreater(poisson_probs(3, .5)[0], .8)

    def test_future_and_same_day_excluded(self):
        data = fixture()
        cutoff = data[10].day
        original = PoissonModel().fit(data, cutoff).predict('A', 'B')
        changed = [m if m.day < cutoff else Match(m.day, m.home, m.away, 20, 0, m.odds) for m in data]
        self.assertEqual(original, PoissonModel().fit(changed, cutoff).predict('A', 'B'))

    def test_walk_forward_no_leakage(self):
        result = backtest(fixture(), min_train=10)
        self.assertEqual(result['predictions'], 30)
        self.assertTrue(all(r['training_end'] < r['date'] for r in result['ledger']))

    def test_virtual_accounting_and_daily_exposure(self):
        result = backtest(fixture(), min_train=10, min_edge=0)
        self.assertGreater(result['virtual_bets'], 0)
        self.assertAlmostEqual(result['final_virtual_balance'], 1000 + sum(r['virtual_profit'] for r in result['ledger']))
        self.assertGreaterEqual(result['final_virtual_balance'], 0)
        self.assertEqual(result['mode'], 'VIRTUAL_ONLY')
        for day in {r['date'] for r in result['ledger']}:
            stakes = [r['virtual_stake'] for r in result['ledger'] if r['date'] == day and r['selection'] is not None]
            self.assertLessEqual(len(set(stakes)), 1)

    def test_missing_odds_still_scores(self):
        rows = [Match(m.day, m.home, m.away, m.hg, m.ag) for m in fixture()]
        result = backtest(rows, min_train=10)
        self.assertEqual(result['virtual_bets'], 0)
        self.assertIsNone(result['virtual_roi'])
        self.assertEqual(result['final_virtual_balance'], 1000.)
        self.assertTrue(math.isfinite(result['log_loss']))

    def test_future_does_not_change_earlier_predictions(self):
        data = fixture()
        first = backtest(data[:20], min_train=10)['ledger']
        extended = backtest(data, min_train=10)['ledger']
        self.assertEqual(first, extended[:len(first)])

    def test_unknown_teams_and_untrained(self):
        model = PoissonModel()
        with self.assertRaises(ValueError):
            model.predict('X', 'Y')
        self.assertAlmostEqual(sum(model.fit(fixture(), date(2025,1,1)).predict('X','Y')), 1.)

    def test_csv_formats_duplicates_and_invalid_scores(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'matches.csv'
            header = 'Date,HomeTeam,AwayTeam,FTHG,FTAG,B365H,B365D,B365A\n'
            for dt in ('01/02/2024', '01/02/24', '2024-02-01'):
                path.write_text(header + dt + ',A,B,2,1,2,3,4\n')
                self.assertEqual(load_matches(path)[0].day, date(2024,2,1))
            path.write_text(header + '01/02/24,A,B,2,1,2,3,4\n' * 2)
            with self.assertRaises(ValueError):
                load_matches(path)
            path.write_text(header + '01/02/24,A,B,-1,1,2,3,4\n')
            with self.assertRaises(ValueError):
                load_matches(path)

    def test_invalid_risk_parameters(self):
        for kwargs in ({'initial': -1}, {'stake_fraction': .1}, {'min_train': 0}, {'min_edge': float('nan')}):
            with self.assertRaises(ValueError):
                backtest(fixture(), **kwargs)


if __name__ == '__main__':
    unittest.main()
