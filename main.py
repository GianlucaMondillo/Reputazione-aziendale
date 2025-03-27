import os
import sys
import json
import pandas as pd
import matplotlib.pyplot as plt
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification # Per caricare modello post-training

# --- Aggiungi la root del progetto al PYTHONPATH ---
# Questo permette import tipo 'from src.config import ...'
# Assumiamo che questo script main.py sia nella root del progetto.
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
print(f"Project root '{project_root}' aggiunto al PYTHONPATH.")

# --- Importa moduli locali ---
try:
    from src.config import * # Importa tutte le configurazioni
    from src.preprocessing import preprocess_text # Singola funzione
    from src.model_utils import (
        load_model_and_tokenizer,
        prepare_dataset,
        train_model
    )
    from src.monitoring import ReputationMonitor # Classe principale
    from src.cicd import CICDPipeline # Classe principale
    from src.documentation import generate_documentation # Funzione
except ImportError as e:
    print(f"[ERRORE CRITICO] Import moduli fallito: {e}")
    print("Verifica che tutti i file .py siano stati creati correttamente in src/.")
    sys.exit(1)

def main_workflow():
    """Orchestra l'intero processo."""
    print("\n" + "="*20 + " AVVIO WORKFLOW COMPLETO " + "="*20)

    # --- Definisci Directory di Output Base ---
    # Relativa alla root del progetto dove si trova main.py
    # Questa directory sarà ignorata da Git.
    output_dir_base = os.path.join(project_root, "output")
    # Verrà creata da train_model o altre funzioni se necessario.
    print(f"Directory di Output Base (ignorata da Git): {output_dir_base}")
    # Crea la directory base esplicitamente per sicurezza, anche se Trainer dovrebbe farlo
    os.makedirs(output_dir_base, exist_ok=True)

    # --- [FASE 1-2] Caricamento Modello Base e Preparazione Dataset ---
    print("\n" + "-"*10 + " FASE 1 & 2: Modello Base e Dataset " + "-"*10)
    final_model_path, eval_results = None, None # Inizializza
    try:
        model, tokenizer = load_model_and_tokenizer(MODEL_NAME)
        # Decidi se usare il dataset alternativo (es. per velocità o se il download fallisce)
        force_alternative_dataset = False # Metti True per forzare l'alternativo
        train_ds, val_ds = prepare_dataset(tokenizer, use_alternative=force_alternative_dataset)
    except Exception as e:
        print(f"\n[ERRORE CRITICO] Fase 1/2 (Modello/Dataset) fallita: {e}")
        sys.exit(1) # Interrompi se non possiamo caricare/preparare dati

    # --- [FASE 3] Fine-tuning Modello ---
    print("\n" + "-"*10 + " FASE 3: Fine-Tuning Modello " + "-"*10)
    try:
        # Passa la directory base dove verranno create 'checkpoints' e 'fine_tuned_model'
        final_model_path, eval_results = train_model(
            model, tokenizer, train_ds, val_ds, output_dir_base
        )
        # Assicurati che le metriche siano un dizionario
        eval_results = eval_results if isinstance(eval_results, dict) else {}
        print(f"Modello fine-tuned salvato in: {final_model_path}")
    except Exception as e:
        print(f"\n[ERRORE CRITICO] Fase 3 (Training) fallita: {e}")
        # Potremmo decidere di continuare con un modello pre-addestrato o fermarci.
        # Per ora, ci fermiamo se il training fallisce.
        sys.exit(1)

    # --- [FASE 4] Setup e Demo Monitoraggio ---
    print("\n" + "-"*10 + " FASE 4: DEMO MONITORAGGIO " + "-"*10)
    monitor = None
    monitor_history = [] # Inizializza vuoto
    try:
        if not final_model_path or not os.path.isdir(final_model_path):
             raise ValueError("Path del modello fine-tuned non valido o mancante.")

        # Inizializza il monitor con il path del modello APPENA addestrato
        monitor = ReputationMonitor(final_model_path)

        # Batch di Esempio per la demo
        demo_batches = {
            "Positivo": (["Servizio eccellente!", "Molto soddisfatto!", "Team fantastico"], "2024-01-10 10:00:00"),
            "Misto": (["Funziona, ma lento.", "Prezzo giusto.", "Interfaccia ok."], "2024-02-15 11:30:00"),
            "Negativo": (["Pessima qualità.", "Non rispondono mai.", "Da evitare assolutamente!"], "2024-03-20 14:00:00"),
            "Critico": (["TRUFFA!", "MAI PIU!", "DENUNCIA!", "Servizio inesistente"], "2024-04-01 09:00:00")
        }
        for name, (texts, ts) in demo_batches.items():
            print(f"\nAnalisi batch demo '{name}'...")
            report = monitor.analyze_batch(texts, timestamp=ts)
            if report:
                monitor.generate_alert(report) # Genera alert se necessario
            else:
                print(f"  Analisi batch '{name}' fallita.")

        # Aggiorna history per documentazione
        monitor_history = monitor.history

        # Visualizza e salva trend (solo se ci sono almeno 2 report)
        if len(monitor_history) >= 2:
             monitor.visualize_trends(output_dir_base)
        else:
             print("Non abbastanza dati per visualizzare i trend.")


        # Esporta metriche (solo se c'è storia)
        if monitor_history:
            monitor.export_metrics(output_dir_base, format="json")
            monitor.export_metrics(output_dir_base, format="csv")
        else:
             print("Nessuna metrica da esportare.")

    except Exception as e:
        print(f"\n[ERRORE] Fase 4 (Monitoraggio Demo) fallita: {e}")
        # Non interrompe il flusso, ma monitor_history rimarrà vuoto o parziale

    # --- [FASE 5] Esecuzione Pipeline CI/CD (Simulata) ---
    print("\n" + "-"*10 + " FASE 5: PIPELINE CI/CD (SIMULATA) " + "-"*10)
    pipeline_report = {"status": "skipped", "reason": "Prerequisiti non soddisfatti (es. training fallito)"}
    if final_model_path and eval_results is not None: # Esegui solo se training OK
        try:
            # Inizializza la pipeline CI/CD
            cicd_pipeline = CICDPipeline(
                 model_path=final_model_path,
                 eval_results=eval_results,
                 output_dir_base=output_dir_base
             )
            pipeline_report = cicd_pipeline.run_pipeline()
        except Exception as e:
            print(f"\n[ERRORE CRITICO] Fase 5 (CI/CD) fallita durante l'inizializzazione o esecuzione: {e}")
            pipeline_report = {"status": "error", "error_message": str(e)}
    else:
        print("[SKIP] Pipeline CI/CD saltata perché il training non è stato completato con successo.")


    # --- [FASE 6] Generazione Documentazione Finale ---
    print("\n" + "-"*10 + " FASE 6: GENERAZIONE DOCUMENTAZIONE " + "-"*10)
    docs_dir_project = os.path.join(project_root, "docs") # Path della dir 'docs' versionata
    try:
        generate_documentation(
            output_dir_base=output_dir_base,
            docs_dir_project=docs_dir_project, # Directory dove salvare i .md finali
            eval_results=eval_results if eval_results else {}, # Passa dict vuoto se None
            monitor_history=monitor_history, # Passa la history raccolta
            pipeline_report=pipeline_report # Passa il report della CI/CD
        )
    except Exception as e:
        print(f"\n[ERRORE] Fase 6 (Documentazione) fallita: {e}")


    # --- [FASE 7] Report Riassuntivo Finale ---
    print("\n" + "-"*10 + " FASE 7: REPORT RIASSUNTIVO " + "-"*10)
    # Prepara i dati per il report finale, gestendo possibili None
    final_model_path_rel = os.path.relpath(final_model_path, project_root) if final_model_path else None
    output_dir_rel = os.path.relpath(output_dir_base, project_root)
    docs_dir_rel = os.path.relpath(docs_dir_project, project_root)
    trends_plot_path_rel = os.path.join(output_dir_rel, "sentiment_trends.png") if len(monitor_history) >= 2 else None
    metrics_paths_rel = [os.path.join(output_dir_rel, f) for f in ["sentiment_metrics_history.json", "sentiment_metrics_history.csv"] if monitor_history]

    final_summary_report = {
        "project_name": "Monitoraggio Reputazione Online",
        "execution_timestamp": pd.Timestamp.now().isoformat(),
        "overall_status": pipeline_report.get('status', 'unknown').upper(), # Basato su CI/CD se eseguita
        "model_info": {
            "base_model": MODEL_NAME,
            "finetuned_path": final_model_path_rel, # Path relativo
            "evaluation": eval_results if eval_results else "N/A"
        },
        "monitoring_demo_summary": {
            "batches_analyzed": len(monitor_history),
            "last_status": monitor_history[-1]["sentiment_status"] if monitor_history else "N/A",
            "trends_plot_path": trends_plot_path_rel,
            "metrics_export_paths": metrics_paths_rel
        },
        "cicd_summary": pipeline_report,
        "documentation_path": docs_dir_rel, # Path relativo dir docs/
        "output_dir_unversioned": output_dir_rel # Path relativo dir output/
    }

    # Salva il report nella directory di output base
    report_filename = "final_project_summary_report.json"
    report_path_abs = os.path.join(output_dir_base, report_filename)
    try:
        with open(report_path_abs, "w", encoding="utf-8") as f:
            # Usa default=str per gestire tipi non serializzabili come Timestamp o numpy floats
            json.dump(final_summary_report, f, indent=2, ensure_ascii=False, default=str)
        print(f"Report riassuntivo finale salvato in: {os.path.join(output_dir_rel, report_filename)}")
    except Exception as e:
        print(f"[ERRORE] Salvataggio report riassuntivo fallito: {e}")

    print("\n" + "="*20 + " WORKFLOW COMPLETATO " + "="*20)
    print(f"-> Output (modello, reports, etc.) in directory: '{output_dir_rel}' (NON versionata)")
    print(f"-> Documentazione (aggiornata) in directory: '{docs_dir_rel}' (Versionata)")
    print("-> Eseguire la Fase 4 (Commit e Push) per salvare il codice su GitHub.")

# --- Punto di ingresso dello script ---
if __name__ == "__main__":
    main_workflow()
