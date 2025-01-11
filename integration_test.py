import fasttext
import os

def test_integration(model_path="fasttext_model.ftz", test_file="test_temp.txt", min_accuracy=0.3):
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Modello FastText non trovato: {model_path}")

    if not os.path.isfile(test_file):
        raise FileNotFoundError(f"File test non trovato: {test_file}")

    model = fasttext.load_model(model_path)
    result = model.test(test_file)
    n, precision, recall = result
    accuracy = precision  # in classification = precision

    print(f"[Integration Test] Esempi di test: {n}")
    print(f"[Integration Test] Accuracy su test: {accuracy:.4f}")

    if accuracy < min_accuracy:
        raise ValueError(f"Test fallito! Accuracy ({accuracy:.4f}) < soglia minima {min_accuracy}")
    else:
        print(f"Test passato! Accuracy >= {min_accuracy}")

if __name__ == "__main__":
    test_integration()
