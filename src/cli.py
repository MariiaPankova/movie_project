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
from db import DBConnector
from model import get_model
from typing import List
from rich import print

app = typer.Typer()


def get_df(path1: str, path2: str):
    df1 = pd.read_csv(path1)
    df2 = pd.read_csv(path2)
    df1.columns = ["id", "tittle", "cast", "crew"]
    df2 = df2.merge(df1, on="id")
    df2["overview"] = df2["overview"].fillna("")
    # print(df2['overview'].str.len().max())
    return df2


@app.command()
def init_db(df_path1: str, df_path2: str, drop: bool = False):
    """
    Creates new database with movie collection or optionally drops and
    recreates existing with dataset from df_path1 and df_path2 dataframes.

    If --drop is True recreates base.
    """
    df = get_df(df_path1, df_path2)
    model = get_model()
    embeddings = model.encode(df["overview"])
    return DBConnector.init_db(df, embeddings, drop)


@app.command()
def insert(name: str, description: str):
    """
    Inserts a new datapoint to a base with a movie name and description.
    """
    db = DBConnector()
    model = get_model()
    desc_embedded = model.encode([description])
    db.add_to_base([name], [description], desc_embedded)


@app.command()
def search(query: str, k_nearest: int = 5):
    """
    Returns best --k_nearest recomendations for your query.
    """
    db = DBConnector()
    model = get_model()
    query_embedded = model.encode([query])
    print(db.get_nearest(query_embedded, k_nearest))


if __name__ == "__main__":
    app()
    # df = get_df('dataset/tmdb_5000_credits.csv', 'dataset/tmdb_5000_movies.csv')
    # embeddings = get_embeddings(df)
    # movie_collection = DBConnector.recreate_db(df, embeddings)
    # movie_collection = DBConnector()
    # movie_collection.add_to_base(
    #    ["Smallfoot"],
    #    [
    #        "High up on a mountain peak surrounded by clouds, a secret Yeti society lives in peace and harmony. One day, a Yeti witnesses an airplane crash; Inside lies 'Smallfoot', a legendary creature that will rock the society to its core."
    #    ],
    # )
    # nearest = movie_collection.get_nearest(
    #     ["secret Yeti society", "High in the mountains"], 2
    # )
    # print(nearest)
