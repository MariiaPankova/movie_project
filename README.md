# Movie recomendation system

A python [FastAPI](https://fastapi.tiangolo.com/) service for content-based movie recomendational system based on [SBERT](https://www.sbert.net/) using [Milvus](https://milvus.io/) vector database. 

Features:
1. Finds the best matching movie for your text desctiption.
2. Adds a new movie to a DB.

 ## Quickstart:
 1. Download dataset from [kaggle](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata) and extract to `dataset` folder.
 2. Run  ```docker compose up -d --build``` to start API and DB.
    1. See API logs with ```docker compose logs movie_api```
 3. You can see API docs on http://localhost:8000/docs
 4. To populate DB with dataset instances run 
   ```db_host=localhost python src/cli.py init-db 'dataset/tmdb_5000_credits.csv' 'dataset/tmdb_5000_movies.csv' --drop```

Read CLI docs using ```python src/cli.py --help```