import typer
import pandas as pd
import sentence_transformers
import numpy as np
from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)

from .core.db import DBConnector
from typing import List, Union
from rich import print
from fastapi import FastAPI
from contextlib import asynccontextmanager
from .core import models
from .core.model import get_model
import traceback
import pymilvus.exceptions as e

db = None
model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global db, model
    try:
        db = DBConnector()
    except e.SchemaNotReadyException:
        db = DBConnector.init_db() 
    model = get_model()
    yield


app = FastAPI(lifespan=lifespan)


@app.put("/movie")
def insert(item: models.MovieItem):
    """
    Inserts a new datapoint to a base with a movie name and description.
    """
    try:
        desc_embedded = model.encode([item.description])
        db.add_to_base([item.name], [item.description], desc_embedded)
        return {"STATUS": "OK"}

    except Exception as e:
        return {"STATUS": "ERROR", "MESSAGE": str(e), "DETAIL": traceback.format_exc()}


@app.get("/movie")
def search(desctiption: str, k_nearest: int = 5) -> list[models.MovieItem] | dict:
    """
    Returns best --k_nearest recomendations for your description.
    """
    try:
        query_embedded = model.encode([desctiption])
        db_nearest = db.get_nearest(query_embedded, k_nearest)
        return [
            models.MovieItem(name=name, description=desc)
            for name, desc in db_nearest.items()
        ]
    except Exception as e:
        return {"STATUS": "ERROR", "MESSAGE": str(e), "DETAIL": traceback.format_exc()}
