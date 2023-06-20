FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY src/ src/
COPY .env .env
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
