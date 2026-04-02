FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for OpenCV
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency spec first for layer caching
COPY pyproject.toml ./

# Copy source code
COPY src/ ./src/
COPY detect.py app.py api.py evaluate.py ./
COPY test_images/ ./test_images/
COPY .env.example ./

# Install Python dependencies
RUN pip install --no-cache-dir -e ".[all]"

# Expose ports for Gradio UI (7860) and FastAPI (8000)
EXPOSE 7860 8000

# Default: run the Gradio UI
CMD ["python", "app.py"]
