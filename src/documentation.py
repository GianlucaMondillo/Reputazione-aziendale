import os
import pandas as pd

# Importa costanti necessarie da config
from .config import MODEL_NAME, CI_CD_REPO_NAME, MONITORING_AVG_SENTIMENT_THRESHOLDS

def generate_documentation(output_dir_base, docs_dir_project, eval_results, monitor_history, pipeline_report):
    """Genera i file di documentazione del progetto."""
    print("\n[+] Generazione documentazione...")

    # Assicurati che le directory esistano
    docs_dir_artifact = os.path.join(output_dir_base, "docs_generated")
    os.makedirs(docs_dir_project, exist_ok=True) # Dir versionata
    os.makedirs(docs_dir_artifact, exist_ok=True) # Dir artefatto in output/

    # Helper per ottenere metriche e status (gestendo None o dict vuoti)
    eval_results = eval_results or {}
    pipeline_report = pipeline_report or {}
    monitor_history = monitor_history or []

    eval_acc = eval_results.get('eval_accuracy', 0.0)
    eval_f1 = eval_results.get('eval_f1', 0.0)
    eval_prec = eval_results.get('eval_precision', 0.0)
    eval_rec = eval_results.get('eval_recall', 0.0)
    pipeline_status = pipeline_report.get('status', 'N/A')
    last_monitor_status = monitor_history[-1]['sentiment_status'] if monitor_history else "N/A"

    # 1. Documentazione Progetto (project_documentation.md)
    project_docs_content = f"""# Progetto Monitoraggio Reputazione Online - Documentazione Tecnica

## 1. Introduzione
Questo documento descrive l'architettura e l'implementazione del sistema di monitoraggio della reputazione online, sviluppato utilizzando Python e la libreria `transformers`. Il sistema analizza il sentiment dei testi utilizzando un modello RoBERTa fine-tuned.

## 2. Architettura del Sistema

### 2.1 Modello di Analisi del Sentiment
- **Modello Base**: `{MODEL_NAME}` (RoBERTa pre-addestrato su dati Twitter).
- **Fine-tuning**: Eseguito su dataset `tweet_eval` (configurazione `sentiment`) o su un dataset alternativo incorporato se il primo non è disponibile. Il dataset viene ridotto per scopi dimostrativi.
- **Classi di Sentiment**: `negativo` (0), `neutro` (1), `positivo` (2).
- **Metriche Chiave (Validation Set)**:
    - Accuracy: `{eval_acc:.4f}`
    - F1 (weighted): `{eval_f1:.4f}`
    - Precision (weighted): `{eval_prec:.4f}`
    - Recall (weighted): `{eval_rec:.4f}`
    *(Nota: le metriche si riferiscono all'ultima esecuzione del training).*

### 2.2 Sistema di Monitoraggio (`src/monitoring.py`)
- Classe `ReputationMonitor` che incapsula la logica di analisi.
- Metodo `analyze_batch`: Preprocessa testi, esegue la pipeline di sentiment, calcola metriche aggregate (distribuzione, sentiment medio), determina lo status della reputazione.
- Metodo `visualize_trends`: Genera e salva un grafico temporale della distribuzione del sentiment e del sentiment medio.
- Metodo `generate_alert`: Crea un dizionario di alert se lo status è `Preoccupante` o `Critico`.
- Metodo `export_metrics`: Salva lo storico delle metriche aggregate in formato JSON o CSV.

### 2.3 Pipeline CI/CD (`src/cicd.py`) (Simulata)
- Classe `CICDPipeline` che simula un workflow MLOps.
- **Stages Simulati**:
    1.  `Code Quality`: Controlli statici (linting, style).
    2.  `Unit Tests`: Test su funzioni isolate (preprocessing, interpretazione label).
    3.  `Integration Tests`: Test sul modello fine-tuned con casi d'uso specifici per sentiment.
    4.  `Prepare Deployment`: Creazione pacchetto per Hugging Face Hub (modello, tokenizer, README).
    5.  `Simulate Deployment`: Simulazione dell'upload su HF Hub.
    6.  `Setup Monitoring`: Creazione configurazione per monitoraggio post-deploy.
- **Stato Ultima Esecuzione Pipeline**: `{pipeline_status.upper()}` (Vedi `output/artifacts/pipeline_summary_report.json` per dettagli).

## 3. Risultati dell'Esecuzione Corrente
- Modello fine-tuned salvato in `output/fine_tuned_model/`.
- Batch analizzati nella demo: {len(monitor_history)}.
- Ultimo status monitoraggio demo: `{last_monitor_status}`.
- Artefatti CI/CD (report, deploy package) in `output/artifacts/`.

## 4. Setup ed Esecuzione
Vedi `README.md` per le istruzioni su come clonare, installare le dipendenze ed eseguire `main.py`.

## 5. Possibili Miglioramenti
- Integrazione con sorgenti dati reali (API social, feed RSS, etc.).
- Creazione di una dashboard interattiva (es. Streamlit, Gradio).
- Aggiunta di analisi più profonde (es. topic modeling sui commenti negativi).
- Implementazione di un ciclo di re-training automatico (MLOps completo).

*(Documento generato automaticamente il {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')})*
"""
    # Salva in entrambe le posizioni
    doc_proj_path_repo = os.path.join(docs_dir_project, "project_documentation.md")
    doc_proj_path_artifact = os.path.join(docs_dir_artifact, "project_documentation.md")
    try:
        with open(doc_proj_path_repo, "w", encoding="utf-8") as f: f.write(project_docs_content)
        # Copia anche nella directory degli artefatti
        shutil.copy2(doc_proj_path_repo, doc_proj_path_artifact)
        print(f"- Documentazione tecnica salvata in '{docs_dir_project}' e '{docs_dir_artifact}'.")
    except Exception as e: print(f"[ERRORE] Salvataggio documentazione tecnica fallito: {e}")

    # 2. Guida Utente (user_guide.md)
    # Usa le soglie da config
    thresh_exc = MONITORING_AVG_SENTIMENT_THRESHOLDS.get('Eccellente', 0.5)
    thresh_pos = MONITORING_AVG_SENTIMENT_THRESHOLDS.get('Positivo', 0.2)
    thresh_neu = MONITORING_AVG_SENTIMENT_THRESHOLDS.get('Neutro', -0.2)
    thresh_pre = MONITORING_AVG_SENTIMENT_THRESHOLDS.get('Preoccupante', -0.5)

    user_guide_content = f"""# Guida Utente - Sistema Monitoraggio Reputazione

## 1. Introduzione
Questo sistema permette di analizzare il sentiment di testi (come menzioni sui social media o recensioni) per monitorare la reputazione online di un brand, prodotto o servizio.

## 2. Come Eseguire il Sistema
L'intero processo è orchestrato dallo script `main.py`. Per eseguirlo:
1. Assicurati di aver installato le dipendenze da `requirements.txt`.
2. Esegui `python main.py` dalla directory principale del progetto.

Lo script eseguirà le seguenti fasi:
- Caricamento/Preparazione dati.
- Fine-tuning del modello di sentiment.
- Demo del sistema di monitoraggio su alcuni batch di esempio.
- Simulazione di una pipeline CI/CD (test, preparazione deploy).
- Generazione di questa documentazione.

I principali output verranno salvati nella directory `output/` (che non è tracciata da Git) e la documentazione aggiornata in `docs/`.

## 3. Funzionalità del Monitoraggio

Il cuore del monitoraggio è la classe `ReputationMonitor` in `src/monitoring.py`.

### 3.1 Analizzare un Batch di Testi
Puoi analizzare una lista di stringhe per ottenere un report sul sentiment.


from src.monitoring import ReputationMonitor
# Assumi che 'monitor' sia un'istanza inizializzata correttamente
# monitor = ReputationMonitor("output/fine_tuned_model")

texts_to_check = ["È fantastico!", "Non funziona.", "Così così."]
report = monitor.analyze_batch(texts_to_check)

if report:
    print(f"Status del batch: {report['sentiment_status']}")
    print(f"Sentiment medio: {report['average_sentiment']:.3f}")
    print("Distribuzione:")
    print(f"  Positivo: {report['sentiment_distribution']['positive']['percentage']:.1f}%")
    print(f"  Neutro: {report['sentiment_distribution']['neutral']['percentage']:.1f}%")
    print(f"  Negativo: {report['sentiment_distribution']['negative']['percentage']:.1f}%")

3.2 Visualizzare i Trend
Dopo aver analizzato più batch (la demo in main.py lo fa), puoi generare un grafico dei trend.

# Salva il grafico nella directory 'output'
# Il path del file generato viene restituito
plot_file = monitor.visualize_trends(output_dir_base='output')
if plot_file:
    print(f"Grafico salvato in: {plot_file}")

Il grafico mostra l'evoluzione della distribuzione del sentiment e del sentiment medio nel tempo.

3.3 Generare Alert
Il sistema può generare alert se il sentiment medio scende sotto certe soglie.

# Genera alert basato sull'ultimo report analizzato
alert = monitor.generate_alert(report) # Usa il report da analyze_batch
if alert:
    print(f"ALERT! Severity: {alert['severity']}")
    print(f"Messaggio: {alert['message']}")
    print(f"Esempi negativi: {alert['details']['top_negative_examples']}")
3.4 Esportare le Metriche Storiche
Puoi salvare un riassunto dei report generati nel tempo.

# Salva in formato CSV nella directory 'output'
csv_path = monitor.export_metrics(output_dir_base='output', format='csv')
if csv_path: print(f"Metriche CSV esportate in: {csv_path}")

# Salva in formato JSON nella directory 'output'
json_path = monitor.export_metrics(output_dir_base='output', format='json')
if json_path: print(f"Metriche JSON esportate in: {json_path}")
Use code with caution.
Python
4. Interpretazione dello Status di Sentiment
Lo status viene calcolato in base al sentiment medio del batch analizzato:

Eccellente: >= {thresh_exc:.2f}

Positivo: >= {thresh_pos:.2f}

Neutro: >= {thresh_neu:.2f}

Preoccupante: >= {thresh_pre:.2f}

Critico: < {thresh_pre:.2f}

(Guida generata il {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')})
"""
# Salva in entrambe le posizioni
guide_path_repo = os.path.join(docs_dir_project, "user_guide.md")
guide_path_artifact = os.path.join(docs_dir_artifact, "user_guide.md")
try:
with open(guide_path_repo, "w", encoding="utf-8") as f: f.write(user_guide_content)
# Copia anche nella directory degli artefatti
shutil.copy2(guide_path_repo, guide_path_artifact)
print(f"- Guida utente salvata in '{docs_dir_project}' e '{docs_dir_artifact}'.")
except Exception as e: print(f"[ERRORE] Salvataggio guida utente fallito: {e}")

print("Generazione documentazione completata.")
return docs_dir_project # Ritorna path della dir versionata
