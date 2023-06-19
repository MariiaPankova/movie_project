from pydantic import BaseModel

class MovieItem(BaseModel):
    name: str
    description: str

