import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from datasets import Dataset, load_dataset
import random

# Importa configurazioni dal modulo config
from .config import MODEL_NAME, ID2LABEL, LABEL2ID, MAX_SEQ_LENGTH, BATCH_SIZE, EPOCHS, LEARNING_RATE, WEIGHT_DECAY
# Importa preprocessing dal modulo preprocessing
from .preprocessing import preprocess_text


def load_model_and_tokenizer(model_name=MODEL_NAME):
    """Carica modello e tokenizer pre-addestrati."""
    print(f"\n[+] Caricamento modello '{model_name}'...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(ID2LABEL),
        id2label=ID2LABEL,
        label2id=LABEL2ID
    )
    return model, tokenizer

def tokenize_function(examples, tokenizer):
    """Tokenizza gli esempi di testo."""
    processed_texts = [preprocess_text(text) for text in examples["text"]]
    return tokenizer(
        processed_texts,
        padding="max_length",
        truncation=True,
        max_length=MAX_SEQ_LENGTH
    )

def prepare_dataset(tokenizer, use_alternative=False):
    """Carica, preprocessa e tokenizza il dataset."""
    print("\n[+] Preparazione dataset...")
    if not use_alternative:
        try:
            dataset = load_dataset("tweet_eval", "sentiment")
            print(f"Dataset pubblico caricato: {len(dataset['train'])} train, {len(dataset['test'])} test")

            # Split train/validation
            train_val = dataset["train"].train_test_split(test_size=0.2, seed=42, stratify_by_column="label")
            train_ds = train_val["train"]
            val_ds = train_val["test"]
            # Il dataset test originale può essere usato per una valutazione finale separata se necessario
            # test_ds = dataset["test"]

            # Tokenizzazione
            tokenized_train = train_ds.map(lambda x: tokenize_function(x, tokenizer), batched=True, remove_columns=["text"])
            tokenized_val = val_ds.map(lambda x: tokenize_function(x, tokenizer), batched=True, remove_columns=["text"])

            # Limitiamo il dataset per velocità (opzionale, ma utile per demo)
            train_indices = []
            num_samples_per_class = 50 # Riduci ulteriormente se necessario
            for label in ID2LABEL.keys():
                label_indices = [i for i, lab in enumerate(tokenized_train["label"]) if lab == label]
                sampled_indices = random.sample(label_indices, min(num_samples_per_class, len(label_indices)))
                train_indices.extend(sampled_indices)
            random.shuffle(train_indices)

            tokenized_train = tokenized_train.select(train_indices)
            # Limitiamo anche la validazione
            val_sample_size = min(90, len(tokenized_val))
            tokenized_val = tokenized_val.select(range(val_sample_size))

            # Rinomina colonna 'label' in 'labels' se necessario
            if "label" in tokenized_train.column_names and "labels" not in tokenized_train.column_names:
                tokenized_train = tokenized_train.rename_column("label", "labels")
            if "label" in tokenized_val.column_names and "labels" not in tokenized_val.column_names:
                 tokenized_val = tokenized_val.rename_column("label", "labels")

            print(f"Dataset ridotto: {len(tokenized_train)} train, {len(tokenized_val)} val")
            return tokenized_train, tokenized_val

        except Exception as e:
            print(f"Errore caricamento dataset pubblico ({e}). Utilizzo dataset alternativo.")
            use_alternative = True # Forza l'uso dell'alternativo

    # Dataset alternativo (se use_alternative=True o se il caricamento fallisce)
    print("Utilizzo dataset alternativo...")
    texts = [
        "Servizio eccellente, complimenti!", "Ottimo supporto tecnico", "Prodotto fantastico, lo consiglio vivamente a tutti!",
        "Servizio clienti rapido e risolutivo.", "Davvero soddisfatto dell'acquisto.",
        "Servizio nella media, niente di eccezionale", "Funziona come previsto, ok.", "Interfaccia standard, non male.",
        "Prezzo adeguato alla qualità offerta.", "Nessun problema particolare da segnalare.",
        "Servizio pessimo, non funziona quasi mai", "Supporto inesistente, attese infinite.", "Esperienza terribile, da evitare assolutamente.",
        "Prodotto difettoso e assistenza lenta.", "Molto deluso, non comprerò più."
    ]
    labels = [2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0] # 0=neg, 1=neu, 2=pos

    # Tokenizza e crea Dataset
    encodings = tokenizer(
        [preprocess_text(t) for t in texts], # Preprocessa anche qui
        padding="max_length",
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        return_tensors="pt" # Utile se si usa direttamente torch, ma Dataset.from_dict va bene
    )
    dataset_dict = {
        "input_ids": encodings["input_ids"].tolist(),
        "attention_mask": encodings["attention_mask"].tolist(),
        "labels": labels
    }
    full_dataset = Dataset.from_dict(dataset_dict)

    # Split fittizio per avere train/val (nell'alternativo usiamo lo stesso per semplicità)
    # In un caso reale, avresti dati separati o faresti uno split
    if len(full_dataset) > 1:
      train_val_split = full_dataset.train_test_split(test_size=0.3, seed=42) # 70% train, 30% val
      tokenized_train = train_val_split['train']
      tokenized_val = train_val_split['test']
    else: # Se c'è solo 1 campione, usalo per entrambi per evitare errori
      tokenized_train = full_dataset
      tokenized_val = full_dataset

    print(f"Dataset alternativo: {len(tokenized_train)} train, {len(tokenized_val)} val")
    return tokenized_train, tokenized_val


def compute_metrics(eval_pred):
    """Calcola metriche di valutazione."""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    # Usiamo zero_division=0 per evitare warning/errori se una classe non ha predizioni/label
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "f1": float(f1_score(labels, predictions, average='weighted', zero_division=0)),
        "precision": float(precision_score(labels, predictions, average='weighted', zero_division=0)),
        "recall": float(recall_score(labels, predictions, average='weighted', zero_division=0))
    }

def train_model(model, tokenizer, train_dataset, eval_dataset, output_dir):
    """Esegue il fine-tuning del modello."""
    print("\n[+] Fine-tuning del modello...")
    training_args = TrainingArguments(
        output_dir=os.path.join(output_dir, "checkpoints"),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        evaluation_strategy="epoch",
        save_strategy="epoch", # Salva alla fine di ogni epoca
        load_best_model_at_end=True, # Carica il miglior modello alla fine
        metric_for_best_model="f1", # Usa F1 score per decidere il migliore
        greater_is_better=True, # F1 più alto è meglio
        logging_dir=os.path.join(output_dir, 'logs'),
        logging_steps=10, # Logga ogni 10 step
        report_to="none" # Disabilita report a W&B/TensorBoard di default
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer # Passare il tokenizer è buona pratica
    )

    # Esegui training
    trainer.train()

    # Valutazione finale sul set di validazione col miglior modello
    print("\n[+] Valutazione finale sul set di validazione...")
    eval_results = trainer.evaluate()
    print(f"Metriche dopo fine-tuning:")
    for metric, value in eval_results.items():
        # Stampa solo le metriche di valutazione effettive
        if metric.startswith('eval_'):
            print(f"- {metric[5:]}: {value:.4f}")

    # Salvataggio modello e tokenizer
    model_path = os.path.join(output_dir, "fine_tuned_model")
    trainer.save_model(model_path)
    tokenizer.save_pretrained(model_path)
    print(f"Modello fine-tuned e tokenizer salvati in: {model_path}")

    return model_path, eval_results
