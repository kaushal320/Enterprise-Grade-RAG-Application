FROM python:3.12-slim-bookworm

WORKDIR /app

RUN pip install --no-cache-dir \
    streamlit==1.54.0 \
    requests==2.33.1 \
    "logfire[requests]==4.32.1" \
    python-dotenv==1.2.2

COPY . ./ui/

# Honor dynamic $PORT environment variable set by cloud host (defaults to 8501)
CMD sh -c "streamlit run ui/app.py --server.address=0.0.0.0 --server.port=${PORT:-8501}"
