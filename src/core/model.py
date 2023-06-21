import pandas as pd
import sentence_transformers
from .settings import app_settings


def get_model():
    return sentence_transformers.SentenceTransformer(app_settings.model_name)
