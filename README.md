# Monitoraggio Reputazione Online - Implementazione Completa

Questo progetto implementa un sistema completo per il monitoraggio della reputazione online utilizzando l'analisi del sentiment basata su modelli Transformer (RoBERTa).

Include:
- Fine-tuning di un modello RoBERTa pre-addestrato (`cardiffnlp/twitter-roberta-base-sentiment-latest`) su dati italiani (simulato con `tweet_eval`).
- Un sistema di monitoraggio (`ReputationMonitor`) per analizzare batch di testi, generare report, visualizzare trend e creare alert.
- Una pipeline CI/CD simulata (`CICDPipeline`) che include controlli di qualità, unit/integration testing, preparazione per il deployment (Hugging Face Hub) e configurazione del monitoraggio post-deploy.
- Generazione automatica di documentazione di progetto e guida utente.

## Struttura del Progetto

- `src/`: Contiene il codice sorgente modulare.
  - `config.py`: Costanti e configurazioni.
  - `preprocessing.py`: Funzioni di preprocessing del testo.
  - `model_utils.py`: Funzioni per caricamento modello/tokenizer e training.
  - `monitoring.py`: Classe `ReputationMonitor` e funzioni correlate.
  - `cicd.py`: Classe `CICDPipeline`.
- `main.py`: Script principale per eseguire l'intero workflow.
- `requirements.txt`: Dipendenze del progetto.
- `.gitignore`: File e directory da ignorare in Git.
- `docs/`: Contiene la documentazione generata (user guide, project docs).
- `output/`: (Ignorato da Git) Directory per artefatti generati durante l'esecuzione (modelli, report JSON, plot, etc.).

## Setup

1.  **Clona il repository:**
    ```bash
    git clone https://github.com/TUO_USERNAME/reputation-monitoring-it.git
    cd reputation-monitoring-it
    ```
2.  **Crea un ambiente virtuale (consigliato):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Su Windows: venv\Scripts\activate
    ```
3.  **Installa le dipendenze:**
    ```bash
    pip install -r requirements.txt
    ```

## Esecuzione

Esegui lo script principale per avviare l'intero processo (training, monitoraggio demo, CI/CD):
```bash
python main.py
