import fasttext
import os
from datasets import load_dataset

def main():
    # 1) Carica il dataset
    dataset = load_dataset("tweet_eval", "sentiment")
    train_data = dataset["train"]
    val_data   = dataset["validation"]

    # 2) Crea i file .txt in stile FastText: __label__<class> <testo>
    train_file = "train.txt"
    val_file   = "valid.txt"

    with open(train_file, "w", encoding="utf-8") as f:
        for ex in train_data:
            label = ex["label"]
            text = ex["text"].replace("\n", " ")
            f.write(f"__label__{label} {text}\n")

    with open(val_file, "w", encoding="utf-8") as f:
        for ex in val_data:
            label = ex["label"]
            text = ex["text"].replace("\n", " ")
            f.write(f"__label__{label} {text}\n")

    # 3) Allena un modello di classificazione supervisionata con FastText
    model = fasttext.train_supervised(
        input=train_file,
        lr=0.7,
        epoch=5,
        wordNgrams=1,
        dim=100,
        loss='softmax'
    )

    # 4) Test veloce sul validation set
    result = model.test(val_file)
    print(f"[TRAIN] Validation samples: {result[0]}")
    print(f"[TRAIN] Validation accuracy (precision): {result[1]:.4f}")

    # 5) Salva il modello
    model_file = "fasttext_model.ftz"
    model.save_model(model_file)
    print(f"[TRAIN] Modello salvato in: {model_file}")

if __name__ == "__main__":
    main()
