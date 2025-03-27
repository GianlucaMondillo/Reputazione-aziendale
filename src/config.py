import os

# Configurazione Modello
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
ID2LABEL = {0: "negativo", 1: "neutro", 2: "positivo"}
LABEL2ID = {"negativo": 0, "neutro": 1, "positivo": 2}
MAX_SEQ_LENGTH = 128

# Configurazione Training
BATCH_SIZE = 8
EPOCHS = 3 # Riduci per test rapidi se necessario
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01

# Configurazione Output
# Usiamo os.path.join per la compatibilità tra OS
# Assicurati che PROJECT_PATH sia definito nello script principale
# OUTPUT_DIR verrà definito nello script principale in base a PROJECT_PATH
# OUTPUT_DIR_BASE = "./output" # Definito in main.py

# Configurazione CI/CD
CI_CD_REPO_NAME = "machine-innovators/sentiment-analysis-it"
CI_CD_MODEL_VERSION = "1.0.0"

# Configurazione Monitoraggio
MONITORING_ALERT_THRESHOLD_STATUS = ["Preoccupante", "Critico"]
MONITORING_AVG_SENTIMENT_THRESHOLDS = {
    "Eccellente": 0.5,
    "Positivo": 0.2,
    "Neutro": -0.2,
    "Preoccupante": -0.5,
    # "Critico" è tutto ciò che è sotto -0.5
}
