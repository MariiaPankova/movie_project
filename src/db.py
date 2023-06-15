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


class DBConnector:
    def __init__(self) -> None:
        connections.connect("default", host="localhost", port="19530")
        self.collection = Collection("movie_collection")
        self.collection.load()

    @classmethod
    def init_db(cls, df: pd.DataFrame, embeddings: np.ndarray, drop=False):
        connections.connect("default", host="localhost", port="19530")
        if drop:
            utility.drop_collection("movie_collection")

        fields = [
            FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="name", dtype=DataType.VARCHAR, max_length=1024),
            FieldSchema(name="desc", dtype=DataType.VARCHAR, max_length=2048),
            FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=384),
        ]
        schema = CollectionSchema(fields)
        movie_collection = Collection(name="movie_collection", schema=schema)

        entities = [
            df["tittle"].to_numpy(),
            df["overview"].to_numpy(),
            embeddings,
        ]
        insert_result = movie_collection.insert(entities)
        movie_collection.flush()

        index = {
            "index_type": "IVF_FLAT",
            "metric_type": "IP",
            "params": {"nlist": 256},
        }
        movie_collection.create_index("embeddings", index)
        return cls()

    @classmethod
    def recreate_db(cls, df: pd.DataFrame, embeddings: np.ndarray):
        connections.connect("default", host="localhost", port="19530")
        utility.drop_collection("movie_collection")
        return cls.init_db(df, embeddings)

    def get_nearest(self, query_embedded: np.ndarray, k_nearest: int = 5):
        search_params = {"metric_type": "IP", "params": {"nlist": 256}}

        results = self.collection.search(
            query_embedded,
            anns_field="embeddings",
            param=search_params,
            limit=k_nearest,
            output_fields=["name", "desc"],
        )

        return {hit.entity.get("name"): hit.entity.get("desc") for hit in results[0]}

    def add_to_base(self, name: list, desc: list, desc_embedded: np.ndarray):
        entities = [name, desc, desc_embedded]
        self.collection.insert(entities)
