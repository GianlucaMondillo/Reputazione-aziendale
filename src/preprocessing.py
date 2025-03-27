import re

def preprocess_text(text):
    """
    Preprocessa il testo per l'analisi del sentiment.

    Args:
        text (str): Testo da preprocessare

    Returns:
        str: Testo preprocessato
    """
    if not isinstance(text, str): return "" # Gestisce input non stringa
    text = text.lower()
    # Mantieni hashtag e mention come token distinti ma puliti
    text = re.sub(r'#(\w+)', r' hashtag_\1 ', text)
    text = re.sub(r'@(\w+)', r' mention_\1 ', text)
    text = re.sub(r'http\S+|www\S+', ' URL ', text) # Miglior regex per URL
    # Rimuovi caratteri speciali tranne spazi, preserva numeri e lettere
    text = re.sub(r'[^\w\s]', ' ', text)
    # Rimuovi spazi multipli e trim
    return re.sub(r'\s+', ' ', text).strip()
