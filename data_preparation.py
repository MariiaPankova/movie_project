#https://www.sbert.net/examples/applications/semantic-search/README.html

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

#'dataset/tmdb_5000_credits.csv', 'dataset/tmdb_5000_movies.csv'
def get_df(path1, path2):
    df1 = pd.read_csv(path1)
    df2 = pd.read_csv(path2)
    df1.columns = ['id','tittle','cast','crew']
    df2 = df2.merge(df1, on='id')
    df2['overview'] = df2['overview'].fillna('')
    print(df2['overview'].str.len().max())
    return df2

def get_embeddings(df):
    model = sentence_transformers.SentenceTransformer('all-MiniLM-L6-v2')
    return model.encode(df['overview'])

#print(np.linalg.norm(embeddings, axis=1))
#print(embeddings.shape)

def construct_base(df, embeddings):

    connections.connect("default", host="localhost", port="19530")
    fields = [
        FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema(name="name", dtype=DataType.VARCHAR, max_length=1024),
        FieldSchema(name="desc", dtype=DataType.VARCHAR, max_length=2048),
        FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=384)
    ]
    schema = CollectionSchema(fields)
    movie_collection = Collection(name='movie_collection', schema=schema)

    entities = [
        df['id'].to_numpy(),
        df['tittle'].to_numpy(),
        df['overview'].to_numpy(),
        embeddings
    ]

    insert_result = movie_collection.insert(entities)
    movie_collection.flush()

    index = {
        "index_type": "IVF_FLAT",
        "metric_type": "IP",
        "params": {"nlist": 256},
    }
    movie_collection.create_index("embeddings", index)

if __name__ == "__main__":
    df = get_df('dataset/tmdb_5000_credits.csv', 'dataset/tmdb_5000_movies.csv')
    embeddings = get_embeddings(df)
    #construct_base(df, embeddings)