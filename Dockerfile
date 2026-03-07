FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install PyTorch CPU version and other dependencies
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code (src will be mounted as volume)
# COPY src/ ./src/  # Commented out - using volume mount instead
COPY configs/ ./configs/
COPY weights/ ./weights/

# Create output directory
RUN mkdir -p /app/outputs

# Expose API port
EXPOSE 4001

# Run the Flask API
CMD ["python3", "src/api.py"]