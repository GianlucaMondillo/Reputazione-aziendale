import os
import json
import pandas as pd
import shutil # Per copiare file
import random # Per simulazioni
from transformers import pipeline, AutoTokenizer # Per caricare tokenizer in init e test pipeline

# Importa configurazioni, funzioni e classi necessarie
from .config import CI_CD_REPO_NAME, CI_CD_MODEL_VERSION, MODEL_NAME, ID2LABEL, LABEL2ID, LEARNING_RATE, BATCH_SIZE, EPOCHS, WEIGHT_DECAY
from .preprocessing import preprocess_text
from .monitoring import interpret_sentiment_output # Usa la stessa funzione

class CICDPipeline:
    """
    Simula una pipeline CI/CD per test e deploy del modello.
    """

    def __init__(self, model_path, eval_results, output_dir_base):
        """
        Inizializza la pipeline CI/CD.

        Args:
            model_path (str): Path del modello fine-tuned salvato (es. output/fine_tuned_model).
            eval_results (dict): Risultati della valutazione del modello.
            output_dir_base (str): Directory principale di output del progetto (es. ./output).
        """
        if not model_path or not os.path.isdir(model_path):
             raise FileNotFoundError(f"Percorso modello non valido: {model_path}")

        self.model_path = model_path
        try:
            # Carica il tokenizer corrispondente al modello salvato
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        except Exception as e:
             raise ValueError(f"Impossibile caricare tokenizer da {model_path}") from e

        self.eval_results = eval_results if isinstance(eval_results, dict) else {}
        self.output_dir_base = output_dir_base
        # Crea la directory degli artefatti CI/CD se non esiste
        self.artifacts_dir = os.path.join(output_dir_base, "artifacts")
        os.makedirs(self.artifacts_dir, exist_ok=True)

        self.repo_name = CI_CD_REPO_NAME
        self.version = CI_CD_MODEL_VERSION

        # Crea una pipeline di sentiment per i test di integrazione
        try:
             print(f"[CI/CD] Creazione pipeline di test da: {model_path}")
             # Usa il modello e tokenizer salvati
             self.test_pipeline = pipeline(
                 "sentiment-analysis",
                 model=model_path,
                 tokenizer=self.tokenizer
             )
             print("[CI/CD] Pipeline di test creata.")
        except Exception as e:
             print(f"[CI/CD ERRORE] Impossibile creare pipeline di test: {e}")
             self.test_pipeline = None


    def _save_report(self, data, filename):
        """Salva un report JSON nella directory artifacts."""
        report_path = os.path.join(self.artifacts_dir, filename)
        try:
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"[CI/CD] Report salvato: {filename}")
            return report_path
        except Exception as e:
            print(f"[CI/CD ERRORE] Salvataggio report {filename} fallito: {e}")
            return None

    def run_code_quality_checks(self):
        """Simula controlli di qualità del codice (linting, style)."""
        print("\n[CI/CD Stage 1] Esecuzione controlli qualità codice (Simulato)...")
        checks = {
            "linting (flake8)": random.choice(["passed", "passed", "warning", "passed"]),
            "style (black)": "passed",
            "type_hints (mypy)": random.choice(["passed", "failed", "passed", "passed"]),
            "security (bandit)": "passed",
            "docstrings": "passed"
        }
        self._save_report(checks, "code_quality_report.json")
        failed_checks = [k for k, v in checks.items() if v == "failed"]
        if not failed_checks: return True
        else:
             print(f"[CI/CD] Controlli qualità falliti: {', '.join(failed_checks)}")
             return False

    def run_unit_tests(self):
        """Simula l'esecuzione di unit test."""
        print("\n[CI/CD Stage 2] Esecuzione Unit Test (Simulato)...")
        # Test specifici su funzioni isolate
        test_cases_preprocess = {
            "Test #hashtag @mention http://example.com extra   space": "test hashtag_hashtag mention_mention url extra space",
            "Test con punteggiatura!!! E numeri 123.": "test con punteggiatura e numeri 123",
            "": "", None: "", "Solo testo.": "solo testo" }
        preprocess_results = {}
        for inp, expected in test_cases_preprocess.items():
            actual = preprocess_text(inp)
            test_name = f"preprocess: '{str(inp)[:20]}...'" if inp else "preprocess: ''"
            preprocess_results[test_name] = "passed" if actual == expected else f"failed (got:'{actual}')"

        test_cases_interpret = [
            ({"label": "LABEL_2", "score": 0.9}, ("positivo", 2)),
            ({"label": "0", "score": 0.8}, ("negativo", 0)),
            ({"label": "neutro", "score": 0.5}, ("neutro", 1)),
            ([{"label": "LABEL_1", "score": 0.99}], ("neutro", 1)), # Lista
            ({}, ("neutro", 1)) ] # Vuoto
        interpret_results = {}
        for i, (inp, expected) in enumerate(test_cases_interpret):
             res = interpret_sentiment_output(inp)
             actual = (res[0], res[1]) # Prendi solo sentiment e id
             interpret_results[f"interpret: case {i+1}"] = "passed" if actual == expected else f"failed (got:{actual})"

        test_results = {**preprocess_results, **interpret_results}
        self._save_report(test_results, "unit_test_report.json")
        failed_tests = [k for k, v in test_results.items() if v != "passed"]
        if not failed_tests: return True
        else:
             print(f"[CI/CD] Unit test falliti: {', '.join(failed_tests)}")
             return False

    def run_integration_tests(self):
        """Simula test di integrazione sul modello fine-tuned."""
        print("\n[CI/CD Stage 3] Esecuzione Integration Test (Modello)...")
        if self.test_pipeline is None:
            print("[CI/CD ERRORE] Pipeline di test non disponibile.")
            return False

        test_cases = {
            "positivo": ["Questo è assolutamente fantastico!", "Servizio impeccabile!", "Adoro questo."],
            "neutro": ["Il pacco è arrivato.", "Il prezzo è ok.", "Funziona."],
            "negativo": ["Esperienza terribile.", "Rotto.", "Supporto inutile."] }
        results = {}
        total_correct, total_tests = 0, 0

        for expected_sentiment, texts in test_cases.items():
            processed_texts = [t for t in map(preprocess_text, texts) if t]
            if not processed_texts: continue
            original_map = {preprocess_text(orig): orig for orig in texts if preprocess_text(orig)}

            try:
                predictions = self.test_pipeline(processed_texts)
                cat_correct, cat_results = 0, []
                for i, pred in enumerate(predictions):
                    proc_text = processed_texts[i]
                    orig_text = original_map.get(proc_text, proc_text)
                    actual_sentiment, _, conf = interpret_sentiment_output(pred)
                    correct = (actual_sentiment == expected_sentiment)
                    if correct: cat_correct += 1
                    cat_results.append({ "text": orig_text, "predicted": actual_sentiment, "expected": expected_sentiment, "correct": correct, "confidence": conf })
                results[expected_sentiment] = { "count": len(processed_texts), "correct": cat_correct, "accuracy": cat_correct/len(processed_texts) }
                total_correct += cat_correct
                total_tests += len(processed_texts)
            except Exception as e:
                print(f"[CI/CD Warning] Errore integration test per '{expected_sentiment}': {e}")
                results[expected_sentiment] = {"error": str(e)}

        overall_accuracy = total_correct / total_tests if total_tests > 0 else 0.0
        accuracy_threshold = 0.70 # Soglia per passare
        passed = overall_accuracy >= accuracy_threshold
        integration_report = { "overall_accuracy": overall_accuracy, "threshold": accuracy_threshold, "passed": passed, "details": results }
        self._save_report(integration_report, "integration_test_report.json")
        if not passed: print(f"[CI/CD] Integration test falliti (Accuracy {overall_accuracy:.2f} < {accuracy_threshold})")
        return passed

    def prepare_for_deployment(self):
        """Prepara gli artefatti per il deployment (es. HuggingFace Hub)."""
        print("\n[CI/CD Stage 4] Preparazione Artefatti per Deployment...")
        deploy_dir = os.path.join(self.artifacts_dir, "deploy_package")
        if os.path.exists(deploy_dir): shutil.rmtree(deploy_dir)
        os.makedirs(deploy_dir)

        # 1. Copia modello e tokenizer
        try:
            print(f"Copia modello da '{self.model_path}' a '{deploy_dir}'...")
            # Copia il contenuto della directory model_path in deploy_dir
            for item in os.listdir(self.model_path):
                s = os.path.join(self.model_path, item)
                d = os.path.join(deploy_dir, item)
                if os.path.isdir(s): shutil.copytree(s, d, False, None)
                else: shutil.copy2(s, d)
            print("Modello/tokenizer copiati.")
        except Exception as e:
            print(f"[CI/CD ERRORE FATALE] Copia modello fallita: {e}")
            return None # Fallimento critico

        # 2. Crea README.md per HF Hub
        eval_acc = self.eval_results.get('eval_accuracy', 0.0)
        eval_f1 = self.eval_results.get('eval_f1', 0.0)
        # (Puoi aggiungere precision/recall se disponibili)

        hf_readme_content = f"""---
language: it
license: apache-2.0
tags: [sentiment-analysis, italian, text-classification, roberta]
datasets: [tweet_eval]
metrics: [accuracy, f1]
model-index:
- name: {self.repo_name.split('/')[-1]}
  results:
  - task: {{ type: text-classification }}
    dataset: {{ name: tweet_eval (sentiment validation), type: tweet_eval, config: sentiment, split: validation }}
    metrics:
    - {{ type: accuracy, value: {eval_acc:.4f} }}
    - {{ type: f1, value: {eval_f1:.4f} }}
---
# Modello Sentiment Analysis Italiano: {self.repo_name.split('/')[-1]}

Modello basato su `{MODEL_NAME}` fine-tuned per analisi sentiment in italiano.
Versione: **{self.version}**

## Performance (Validation Set)
- Accuracy: {eval_acc:.4f}
- F1 (weighted): {eval_f1:.4f}

## Uso con `transformers`
```python
from transformers import pipeline
model_id = "{self.repo_name}" # Es: "machine-innovators/sentiment-analysis-it"
sentiment_analyzer = pipeline("sentiment-analysis", model=model_id)
results = sentiment_analyzer(["Che bello!", "Non mi piace.", "Forse."])
print(results)
# Output atteso (label può essere LABEL_X o nome classe):
# [{'label': 'LABEL_2', 'score': ...}, {'label': 'LABEL_0', 'score': ...}, {'label': 'LABEL_1', 'score': ...}]
"""
readme_path = os.path.join(deploy_dir, "README.md")
try:
with open(readme_path, "w", encoding="utf-8") as f: f.write(hf_readme_content)
print("README.md per Hugging Face creato.")
except Exception as e:
print(f"[CI/CD ERRORE] Creazione README.md fallita: {e}")
return None # README è importante

print(f"[CI/CD] Artefatti pronti in: {deploy_dir}")
    return deploy_dir


def simulate_deployment(self, deploy_dir):
    """Simula il deployment su HuggingFace Hub."""
    print(f"\n[CI/CD Stage 5] Simulazione Deployment su Hugging Face Hub ({self.repo_name})...")
    if not deploy_dir or not os.path.isdir(deploy_dir):
         print("[CI/CD ERRORE] Directory di deploy non valida.")
         return {"status": "failed", "reason": "Deploy directory missing"}

    # Simulazione semplice: assume successo
    deploy_status = "success"
    repo_url = f"https://huggingface.co/{self.repo_name}"

    deploy_log = {
        "status": deploy_status,
        "timestamp": pd.Timestamp.now().isoformat(),
        "version": self.version,
        "repo_id": self.repo_name,
        "repo_url": repo_url,
        "deployed_files_count": len([f for f in os.listdir(deploy_dir) if os.path.isfile(os.path.join(deploy_dir, f))])
    }
    self._save_report(deploy_log, "deploy_log.json")
    if deploy_status != "success": print("[CI/CD] Deploy Simulato FALLITO.")
    return deploy_log


def setup_monitoring(self):
    """Simula la configurazione del monitoraggio post-deploy."""
    print("\n[CI/CD Stage 6] Configurazione Monitoraggio Post-Deploy (Simulato)...")
    baseline_acc = self.eval_results.get('eval_accuracy', 0.7)
    monitoring_config = {
        "model_repo_id": self.repo_name, "version": self.version,
        "check_interval_minutes": 30,
        "metrics": [
            {"name": "latency_p95_ms", "threshold": 600, "alert": "above"},
            {"name": "accuracy_vs_baseline", "baseline": baseline_acc, "threshold_delta": -0.08, "alert": "below"}, # Alert se scende di 8 punti %
            {"name": "error_rate_pct", "threshold": 3.0, "alert": "above"} ],
        "alerting": { "channels": ["slack"], "recipients": ["#model-alerts"] }
    }
    self._save_report(monitoring_config, "monitoring_config.json")
    print(f"[CI/CD] Configurazione monitoraggio salvata ({len(monitoring_config['metrics'])} metriche).")
    return monitoring_config


def run_pipeline(self):
    """Esegue l'intera pipeline CI/CD simulata."""
    print("\n" + "="*15 + " AVVIO PIPELINE CI/CD SIMULATA " + "="*15)
    start_time = pd.Timestamp.now()
    stages_status = {}
    pipeline_success = True # Assume successo iniziale

    # Funzione helper per eseguire e loggare uno stage
    def run_stage(stage_name, func, *args):
        nonlocal pipeline_success
        if not pipeline_success: # Salta stage se pipeline già fallita
            stages_status[stage_name] = "skipped"
            print(f"\n[CI/CD Stage {len(stages_status)+1}] {stage_name.upper()} (SKIPPED)")
            return None
        stage_num = len(stages_status) + 1
        print(f"\n[CI/CD Stage {stage_num}] {stage_name.upper()}...")
        try:
            result = func(*args)
            # Determina successo: True se risultato è True o non None/False
            success = bool(result) if not isinstance(result, bool) else result
            stages_status[stage_name] = "passed" if success else "failed"
            print(f"[CI/CD] Stage '{stage_name}' outcome: {stages_status[stage_name].upper()}")
            if not success: pipeline_success = False
            return result
        except Exception as e:
            print(f"[CI/CD ERRORE FATALE] Stage '{stage_name}' fallito con eccezione: {e}")
            stages_status[stage_name] = "error"
            pipeline_success = False
            # Potresti voler loggare il traceback qui
            return None

    # Esecuzione Stages in sequenza
    run_stage("Code Quality", self.run_code_quality_checks)
    run_stage("Unit Tests", self.run_unit_tests)
    run_stage("Integration Tests", self.run_integration_tests)
    deploy_dir = run_stage("Prepare Deployment", self.prepare_for_deployment)
    deploy_log = run_stage("Simulate Deployment", self.simulate_deployment, deploy_dir)
    monitoring_config = run_stage("Setup Monitoring", self.setup_monitoring)

    end_time = pd.Timestamp.now()
    duration = end_time - start_time

    # Report finale della pipeline
    final_status = "success" if pipeline_success else "failed"
    pipeline_report = {
        "pipeline_run_id": f"ci-sim-{start_time.strftime('%Y%m%d%H%M%S')}",
        "status": final_status,
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "duration": str(duration),
        "model_version": self.version,
        "stages": stages_status,
        "deployment_info": deploy_log,
        "monitoring_info": monitoring_config,
        "artifacts_location": self.artifacts_dir
    }
    report_path = self._save_report(pipeline_report, "pipeline_summary_report.json")

    print("\n" + "="*15 + f" PIPELINE CI/CD COMPLETATA (Status: {final_status.upper()}) " + "="*15)
    print(f"Durata: {duration}")
    if report_path: print(f"Report completo pipeline: {report_path}")

    return pipeline_report
