FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements first (for caching)
COPY agent/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy agent code and UI
COPY agent/ ./agent/
COPY ui/ ./ui/
COPY config.yaml .

# Expose Streamlit port
EXPOSE 8501

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV API_BASE_URL=http://api-service:8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8501/_stcore/health').raise_for_status()"

# Run Streamlit
CMD ["streamlit", "run", "ui/chat.py", "--server.port=8501", "--server.address=0.0.0.0"]
