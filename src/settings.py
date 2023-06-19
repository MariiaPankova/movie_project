from pydantic import BaseSettings


class Settings(BaseSettings):
    db_host: str
    db_port: str
    db_alias: str = "default"
    db_collection_name: str = "movie_collection"

    model_name: str = "all-MiniLM-L6-v2"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


app_settings = Settings()
