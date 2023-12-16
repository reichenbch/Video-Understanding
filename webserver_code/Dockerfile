FROM python:3.10

WORKDIR /app

COPY requirements.txt .
COPY ../umap_embed.pkl .
COPY ../bb_cluster.pkl .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "app.py"]