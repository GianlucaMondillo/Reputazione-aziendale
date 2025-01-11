import os
from huggingface_hub import create_repo, Repository

HF_REPO_ID = "TUO_USERNAME/fasttext-sentiment-model"  # <-- Sostituisci con il tuo username
LOCAL_REPO_DIR = "./hf_ft_repo"
MODEL_FILE = "fasttext_model.ftz"

def main():
    # 1) Recupera il token (dai secrets di GitHub Actions)
    hf_token = os.environ.get("HF_TOKEN", None)
    if hf_token is None:
        raise ValueError("Non hai impostato la variabile d'ambiente HF_TOKEN")

    if not os.path.isfile(MODEL_FILE):
        raise FileNotFoundError(f"Modello non trovato: {MODEL_FILE}")

    # 2) Crea (o usa) un repository su Hugging Face
    create_repo(repo_id=HF_REPO_ID, token=hf_token, private=False, exist_ok=True)
    repo = Repository(local_dir=LOCAL_REPO_DIR, clone_from=HF_REPO_ID, use_auth_token=hf_token)

    # 3) Copia il file .ftz nel repository locale
    os.system(f"cp {MODEL_FILE} {LOCAL_REPO_DIR}")

    # 4) Esegui commit e push
    repo.git_add()
    repo.git_commit("Auto-deploy: Update FastText model")
    repo.git_push()

    print("[DEPLOY] Modello caricato su Hugging Face!")

if __name__ == "__main__":
    main()
