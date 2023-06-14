# https://www.sbert.net/examples/applications/semantic-search/README.html

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


def get_df(path1: str, path2: str):
    df1 = pd.read_csv(path1)
    df2 = pd.read_csv(path2)
    df1.columns = ["id", "tittle", "cast", "crew"]
    df2 = df2.merge(df1, on="id")
    df2["overview"] = df2["overview"].fillna("")
    # print(df2['overview'].str.len().max())
    return df2


def init_db(df_path1=None, df_path2=None, drop=False):
    if (df_path1 is None) and (df_path2 is None):
        return DBConnector.init_db(drop)
    df = get_df(df_path1, df_path2)
    embeddings = get_model.encode(df["overview"])
    return DBConnector.init_db(df, embeddings, drop)

    
def insert():
    pass

def search():
    pass

if __name__ == "__main__":
    # df = get_df('dataset/tmdb_5000_credits.csv', 'dataset/tmdb_5000_movies.csv')
    # embeddings = get_embeddings(df)
    # movie_collection = DBConnector.recreate_db(df, embeddings)
    movie_collection = DBConnector()
    # movie_collection.add_to_base(
    #    ["Smallfoot"],
    #    [
    #        "High up on a mountain peak surrounded by clouds, a secret Yeti society lives in peace and harmony. One day, a Yeti witnesses an airplane crash; Inside lies 'Smallfoot', a legendary creature that will rock the society to its core."
    #    ],
    # )
    nearest = movie_collection.get_nearest(
        ["secret Yeti society", "High in the mountains"], 2
    )
    print(nearest)
