FROM python:3.11-slim

# Avoid prompts from debian frontend
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV UNSTRUCTURED_DISABLE_NOTICE=1

# Install system dependencies for Unstructured, OpenCV, Poppler, Tesseract OCR, and libmagic
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    tesseract-ocr \
    libmagic1 \
    libgl1 \
    libglx-mesa0 \
    libglib2.0-0 \
    libgomp1 \
    libsm6 \
    libxext6 \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first for efficient layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code, configuration, and app
COPY config/ config/
COPY src/ src/
COPY scripts/ scripts/
COPY app.py .

# Create data directories
RUN mkdir -p data/docs data/cache data/chroma_db

EXPOSE 8501

# Default command launches the Streamlit application
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
