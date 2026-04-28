FROM python:3.11-slim-bookworm

WORKDIR /app

RUN apt-get update && apt-get upgrade -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    streamlit==1.32.0 \
    kafka-python==2.0.2 \
    boto3 \
    pandas \
    numpy \
    plotly \
    scikit-learn \
    python-dotenv==1.0.0 \
    joblib

COPY src/ ./src/
COPY features/ ./features/

ENV PYTHONPATH=/app:/app/src
ENV DEMO_MODE=true

EXPOSE 8501

CMD ["/bin/sh", "-c", "streamlit run src/dashboard/app.py --server.port=${PORT:-8501} --server.address=0.0.0.0 --server.headless=true --browser.gatherUsageStats=false"]
