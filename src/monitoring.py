import os
import pandas as pd
import matplotlib.pyplot as plt
import json
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer

# Importa configurazioni e funzioni necessarie
from .config import ID2LABEL, LABEL2ID, MONITORING_ALERT_THRESHOLD_STATUS, MONITORING_AVG_SENTIMENT_THRESHOLDS
from .preprocessing import preprocess_text

# FUNZIONE DI INTERPRETAZIONE DELLE ETICHETTE MIGLIORATA
def interpret_sentiment_output(prediction):
    """
    Interpreta l'output della pipeline di sentiment analysis.
    Gestisce diversi formati e usa ID2LABEL/LABEL2ID.

    Args:
        prediction: Risultato della pipeline per un singolo testo (dict) o lista di dict.

    Returns:
        Tuple (sentiment_str, label_id, confidence): Sentiment, ID e score.
        Ritorna ("neutro", 1, 0.0) in caso di errore.
    """
    if isinstance(prediction, list):
        if not prediction: return "neutro", 1, 0.0
        prediction = prediction[0] # Prendi il primo se è una lista

    if not isinstance(prediction, dict):
         return "neutro", 1, 0.0 # Formato non valido

    label = prediction.get("label", "")
    score = float(prediction.get("score", 0.0))

    # Caso 1: Label standard "LABEL_X"
    if isinstance(label, str) and label.startswith("LABEL_"):
        try:
            label_id = int(label.split("_")[1])
            sentiment = ID2LABEL.get(label_id)
            if sentiment:
                return sentiment, label_id, score
        except (ValueError, IndexError):
            pass # Prova altri metodi

    # Caso 2: Label è già una delle nostre stringhe ("negativo", "neutro", "positivo")
    if isinstance(label, str) and label in LABEL2ID:
        label_id = LABEL2ID[label]
        return label, label_id, score

    # Caso 3: Label è un numero (o stringa numerica) che corrisponde a un ID
    try:
        label_id = int(label)
        sentiment = ID2LABEL.get(label_id)
        if sentiment:
            return sentiment, label_id, score
    except (ValueError, TypeError):
         pass # Prova altri metodi

    # Fallback (raro se il modello è configurato bene con id2label/label2id)
    # Potremmo aggiungere una logica basata sullo score, ma è meno affidabile
    print(f"Warning: Impossibile interpretare label '{label}'. Default a 'neutro'.")
    return "neutro", 1, score # Default a neutro

# CLASSE MONITOR REPUTAZIONE
class ReputationMonitor:
    """Sistema di monitoraggio della reputazione online."""

    def __init__(self, model_path_or_pipeline):
        """
        Inizializza il monitor.

        Args:
            model_path_or_pipeline: Path del modello salvato o una pipeline già inizializzata.
        """
        if isinstance(model_path_or_pipeline, str):
            print(f"\n[+] Caricamento modello fine-tuned da {model_path_or_pipeline} per il monitoraggio...")
            try:
                model_finetuned = AutoModelForSequenceClassification.from_pretrained(
                    model_path_or_pipeline,
                    num_labels=len(ID2LABEL),
                    id2label=ID2LABEL,
                    label2id=LABEL2ID
                )
                tokenizer_finetuned = AutoTokenizer.from_pretrained(model_path_or_pipeline)
                self.pipeline = pipeline(
                    "sentiment-analysis",
                    model=model_finetuned,
                    tokenizer=tokenizer_finetuned
                )
                print("[+] Pipeline di sentiment analysis creata con successo.")
            except Exception as e:
                print(f"Errore durante il caricamento del modello/tokenizer da {model_path_or_pipeline}: {e}")
                print("Assicurati che il percorso sia corretto e contenga i file del modello.")
                raise # Rilancia l'eccezione per fermare l'esecuzione se il modello non può essere caricato
        else:
            print("\n[+] Utilizzo pipeline di sentiment analysis fornita.")
            self.pipeline = model_path_or_pipeline # Assume sia una pipeline valida

        self.history = [] # Storico dei report

    def analyze_batch(self, texts, timestamp=None):
        """Analizza un batch di testi e genera un report."""
        if not texts:
            print("Warning: Ricevuto batch vuoto per l'analisi.")
            return None

        # Preprocessing
        processed_texts = [preprocess_text(t) for t in texts]
        # Filtra eventuali testi vuoti dopo il preprocessing
        valid_indices = [i for i, t in enumerate(processed_texts) if t]
        original_texts_valid = [texts[i] for i in valid_indices]
        processed_texts_valid = [processed_texts[i] for i in valid_indices]

        if not processed_texts_valid:
             print("Warning: Nessun testo valido nel batch dopo il preprocessing.")
             return None

        # Predizione (solo sui testi validi)
        try:
            # Imposta return_all_scores=False se vuoi solo la label predetta (più veloce)
            # Ma per confidence serve lo score della label predetta, che è il default
            raw_predictions = self.pipeline(processed_texts_valid)
        except Exception as e:
            print(f"Errore durante l'esecuzione della pipeline di sentiment: {e}")
            return None # Non possiamo continuare senza predizioni

        # Formattazione e Interpretazione
        results = []
        for i, pred in enumerate(raw_predictions):
            original_text = original_texts_valid[i]
            sentiment, label_id, confidence = interpret_sentiment_output(pred)
            results.append({
                "text": original_text, # Testo originale
                "sentiment": sentiment,
                "label_id": label_id,
                "confidence": float(confidence)
            })

        # Conteggio per categoria
        counts = {"positivo": 0, "neutro": 0, "negativo": 0}
        for res in results:
            counts[res["sentiment"]] += 1

        # Calcolo metriche aggregate
        total = len(results)
        sentiment_scores = {"positivo": 1, "neutro": 0, "negativo": -1}
        # Calcola avg_sentiment solo sui risultati validi
        avg_sentiment = sum(sentiment_scores[r["sentiment"]] for r in results) / total if total else 0

        # Determinazione Status basato su soglie definite in config
        status = "Critico" # Default se sotto tutte le soglie
        # Itera sulle soglie dalla più alta alla più bassa
        for s, threshold in sorted(MONITORING_AVG_SENTIMENT_THRESHOLDS.items(), key=lambda item: item[1], reverse=True):
             if avg_sentiment >= threshold:
                 status = s
                 break

        # Creazione del Report
        report_ts = timestamp or pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        report = {
            "timestamp": report_ts,
            "total_mentions_analyzed": total,
            "sentiment_distribution": {
                "positive": {"count": counts["positivo"], "percentage": (counts["positivo"]/total*100) if total else 0},
                "neutral": {"count": counts["neutro"], "percentage": (counts["neutro"]/total*100) if total else 0},
                "negative": {"count": counts["negativo"], "percentage": (counts["negativo"]/total*100) if total else 0}
            },
            "average_sentiment": avg_sentiment,
            "sentiment_status": status,
            "individual_results": results
        }

        self.history.append(report)
        print(f"Report generato per {total} testi ({report_ts}). Status: {status}")
        return report

    def visualize_trends(self, save_path=None):
        """Visualizza i trend di sentiment nel tempo."""
        if not self.history:
            print("Nessun dato storico disponibile per la visualizzazione.")
            return

        timestamps = [pd.to_datetime(r["timestamp"]) for r in self.history]
        # Assicurati che i timestamp siano ordinati per una visualizzazione corretta
        sorted_indices = np.argsort(timestamps)
        timestamps = [timestamps[i] for i in sorted_indices]
        history_sorted = [self.history[i] for i in sorted_indices]


        pos_ratios = [r["sentiment_distribution"]["positive"]["percentage"]/100 for r in history_sorted]
        neu_ratios = [r["sentiment_distribution"]["neutral"]["percentage"]/100 for r in history_sorted]
        neg_ratios = [r["sentiment_distribution"]["negative"]["percentage"]/100 for r in history_sorted]
        avg_sentiments = [r["average_sentiment"] for r in history_sorted]

        plt.style.use('seaborn-v0_8-darkgrid') # Usa uno stile piacevole
        fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True) # Condividi asse x

        # Grafico 1: Distribuzione Sentiment (Stackplot)
        axes[0].stackplot(
            timestamps,
            pos_ratios, neu_ratios, neg_ratios,
            labels=["Positivo", "Neutro", "Negativo"],
            colors=["#4CAF50", "#B0BEC5", "#F44336"], # Colori Material Design
            alpha=0.8
        )
        axes[0].set_title("Evoluzione della Distribuzione del Sentiment", fontsize=14)
        axes[0].set_ylabel("Percentuale (%)", fontsize=12)
        axes[0].legend(loc="upper left")
        axes[0].grid(True, linestyle='--', alpha=0.6)
        axes[0].set_ylim(0, 1) # Percentuale da 0 a 1
        axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y))) # Formatta asse y come %

        # Grafico 2: Sentiment Medio
        axes[1].plot(timestamps, avg_sentiments, marker='o', linestyle='-', color="#673AB7", linewidth=2, label="Sentiment Medio")
        axes[1].axhline(y=0, color="grey", linestyle="--", alpha=0.7, label="Neutro (0.0)")
        # Aggiungi linee per soglie di status per contesto
        for status_name, threshold in MONITORING_AVG_SENTIMENT_THRESHOLDS.items():
             if status_name != "Eccellente": # Non mostrare la soglia più alta
                 axes[1].axhline(y=threshold, color="grey", linestyle=":", alpha=0.5, linewidth=1)
                 # Aggiungi testo vicino alla linea (opzionale, può affollare)
                 # axes[1].text(timestamps[-1], threshold, f' {status_name}', va='center', ha='left', backgroundcolor='white', fontsize=8)

        axes[1].set_title("Evoluzione del Sentiment Medio", fontsize=14)
        axes[1].set_xlabel("Data e Ora", fontsize=12)
        axes[1].set_ylabel("Valore Sentiment (-1 a 1)", fontsize=12)
        axes[1].grid(True, linestyle='--', alpha=0.6)
        axes[1].legend(loc="upper left")
        axes[1].set_ylim(-1.1, 1.1) # Range leggermente esteso

        # Migliora la formattazione delle date sull'asse X
        fig.autofmt_xdate()

        plt.tight_layout(rect=[0, 0.03, 1, 0.97]) # Aggiusta layout e aggiungi spazio per il titolo generale
        fig.suptitle("Monitoraggio Trend Sentiment nel Tempo", fontsize=16, fontweight='bold')

        if save_path:
            try:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Grafico salvato in: {save_path}")
            except Exception as e:
                print(f"Errore durante il salvataggio del grafico: {e}")
        plt.show() # Mostra sempre il grafico

    def generate_alert(self, report):
        """Genera un alert se lo status è preoccupante o critico."""
        if not report: return None

        status = report.get("sentiment_status")
        if status in MONITORING_ALERT_THRESHOLD_STATUS:
            neg_count = report["sentiment_distribution"]["negative"]["count"]
            neg_pct = report["sentiment_distribution"]["negative"]["percentage"]

            # Estrai esempi negativi (ordinati per confidenza decrescente, se disponibile)
            negative_msgs = sorted(
                [r for r in report["individual_results"] if r["sentiment"] == "negativo"],
                key=lambda x: x.get("confidence", 0.0),
                reverse=True
            )
            example_texts = [msg["text"] for msg in negative_msgs[:3]] # Primi 3 testi

            severity = "high" if status == "Critico" else "medium"

            alert = {
                "alert_timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                "report_timestamp": report["timestamp"],
                "severity": severity,
                "message": f"ALERT: Sentiment {status.upper()} rilevato!",
                "details": {
                    "average_sentiment": report["average_sentiment"],
                    "negative_count": neg_count,
                    "negative_percentage": neg_pct,
                    "top_negative_examples": example_texts
                },
                "recommended_action": "Analizzare urgentemente i feedback negativi e considerare azioni correttive."
            }
            print(f"\n*** ALERT GENERATO (Severity: {severity}) ***\n{alert['message']}")
            print(f"   Dettagli: {neg_count} negativi ({neg_pct:.1f}%), Avg Sent: {report['average_sentiment']:.2f}")
            print(f"   Azione: {alert['recommended_action']}")
            return alert

        return None

    def export_metrics(self, output_dir, format="json"):
        """Esporta le metriche aggregate storiche in formato JSON o CSV."""
        if not self.history:
            print("Nessuna metrica storica da esportare.")
            return None

        metrics_data = []
        for report in self.history:
            metrics_data.append({
                "timestamp": report["timestamp"],
                "status": report["sentiment_status"],
                "avg_sentiment": report["average_sentiment"],
                "positive_pct": report["sentiment_distribution"]["positive"]["percentage"],
                "neutral_pct": report["sentiment_distribution"]["neutral"]["percentage"],
                "negative_pct": report["sentiment_distribution"]["negative"]["percentage"],
                "total_mentions": report["total_mentions_analyzed"]
            })

        # Assicura che la directory di output esista
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"sentiment_metrics_history.{format}")

        try:
            if format == "json":
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(metrics_data, f, indent=2, ensure_ascii=False)
            elif format == "csv":
                df = pd.DataFrame(metrics_data)
                df.to_csv(output_path, index=False, encoding="utf-8")
            else:
                print(f"Formato '{format}' non supportato per l'esportazione. Usare 'json' o 'csv'.")
                return None

            print(f"Metriche storiche esportate con successo in: {output_path}")
            return output_path
        except Exception as e:
            print(f"Errore durante l'esportazione delle metriche in {format}: {e}")
            return None
