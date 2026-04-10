FROM python:3.10-slim

# Install pinned dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

WORKDIR /app

EXPOSE 8501

# Cleaned up the redundant flags
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]