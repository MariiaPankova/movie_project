import pandas as pd
import sentence_transformers


def get_model():
    return sentence_transformers.SentenceTransformer("all-MiniLM-L6-v2")
