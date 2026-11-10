# Bankroll virtuale — prima base eseguibile

Solo simulazione: nessun collegamento a bookmaker, nessun pagamento, nessuna operazione con denaro reale. Il modello e una baseline di ricerca, non un sistema di guadagno validato.

## Avvio (Python 3.10 o successivo, nessuna dipendenza esterna)

```sh
python -m unittest discover -v
python bankroll.py storico_serie_a.csv --min-train 100 --output report.json
```

Input CSV nel formato Football-Data: Date, HomeTeam, AwayTeam, FTHG, FTAG. Date accettate: gg/mm/aaaa, gg/mm/aa, aaaa-mm-gg. Quote opzionali B365H, B365D, B365A: tutte e tre presenti o tutte vuote. Il caricamento rifiuta duplicati, punteggi negativi, quote parziali e valori non validi; non inventa risultati mancanti.

## Metodo e controlli

- Poisson indipendenti con medie casa/trasferta e regolarizzazione (5 partite equivalenti).
- Riaddestramento per data: utilizza soltanto partite di giorni precedenti. Nessuna partita dello stesso giorno entra nell'addestramento.
- Quote senza margine ottenute normalizzando le probabilita implicite: sono stime di mercato, non probabilita vere.
- Selezione virtuale quando probabilita modello x quota - 1 supera 0,05. Massimo un esito per partita.
- Saldo iniziale: 1000 unita virtuali. Puntata: 1% del saldo di apertura giornata, con esposizione totale limitata al saldo disponibile. Tutte le puntate del giorno sono fissate prima dei risultati.
- Report JSON con parametri, hash SHA-256 del CSV, probabilita, fine training, esiti, movimenti virtuali, Brier multicategoria (somma), log loss, ROI e drawdown a fine giornata.
- Test con dati sintetici: verificano correttezza tecnica, NON dimostrano accuratezza sportiva o profitto.

## Automazione

GitHub Actions esegue i test a ogni push e pull request; permette anche avvio manuale. Nessun segreto o servizio a pagamento richiesto. I test vengono eseguiti su Python 3.11 e 3.12.

## Limiti e prossimo passo

Non e ancora stato caricato o valutato uno storico reale in questo repository. Occorre aggiungere un CSV verificato, documentarne fonte e data, e avviare la validazione fuori campione. Le quote storiche non hanno un timestamp pre-partita verificato: qualsiasi ROI resta esplorativo. Il modello non considera infortuni, formazioni o dipendenza tra gol. Il Brier di mercato usa solo righe con quote: confrontarlo con quello del modello solo su un campione identico. Parametri da fissare prima della valutazione finale; nessuna ottimizzazione sul periodo di test.

Repository principale: Fanco27/Bankroll-virtuale-.
