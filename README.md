# Bankroll-virtuale-

## Riprodurre il backtest virtuale

Dalla radice del progetto, con Python 3.10 o successivo:

```sh
python -m unittest discover -v
python bankroll.py storico_serie_a.csv --min-train 100 --output report.json
python verify_history.py report.json
```

Il CSV contiene 1.520 incontri di Serie A, 380 per stagione dal 2021/22 al 2024/25. È una proiezione delle otto colonne utilizzate dal modello dal file `data/processed/development_validation.csv` dell'archivio di progetto `Bankroll_archivio_verificato_2021_2026.zip`. Fonte immediata: CSV forniti dall'utente, in formato Football-Data. Provenienza esterna e orario delle quote non verificati indipendentemente. Il test riservato 2025/26 è escluso.

Il report completo comprende le singole previsioni. Il riepilogo in `reports/reference.json` fissa impronta SHA-256, parametri e metriche arrotondate a nove decimali. Il confronto Brier appaiato usa solo incontri con tutte e tre le quote. Un punteggio più basso indica previsioni migliori. Solo unità virtuali. Per i dettagli del modello consultare [GUIDA.md](GUIDA.md).
