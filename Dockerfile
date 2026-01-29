# =============================================================================
# PPI_Predictor_Api Dockerfile
# Automated setup for Protein-Protein Interaction Prediction API
# =============================================================================

FROM python:3.9-slim

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set HuggingFace cache directory
ENV HF_HOME=/app/.cache/huggingface
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    graphviz \
    wget \
    curl \
    build-essential \
    libpq-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create directories for ML models and cache
RUN mkdir -p ml_models/MCAPST5/checkpoints \
    ml_models/MCAPST5/protT5_checkpoint \
    ml_models/MCAPST5/output \
    .cache/huggingface

# Copy requirements first for better caching
COPY requirements.txt .

# Install base Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Install ML dependencies (PyTorch, TensorFlow, Transformers)
# Note: Using CPU versions to reduce image size. For GPU, use cu111 versions.
RUN pip install --no-cache-dir \
    torch==1.9.0+cpu \
    torchvision==0.10.0+cpu \
    torchaudio==0.9.0 \
    -f https://download.pytorch.org/whl/torch_stable.html

RUN pip install --no-cache-dir \
    tensorflow==2.12.0 \
    tensorflow-addons==0.20.0

RUN pip install --no-cache-dir \
    transformers==4.29.2 \
    sentencepiece==0.1.99 \
    h5py==3.8.0 \
    xgboost \
    matplotlib \
    scikit-learn \
    pandas \
    gdown \
    pydot

# Download model checkpoints
RUN echo "Downloading MCAPST5 model checkpoints..." && \
    wget -q --show-progress -O ml_models/MCAPST5/checkpoints/mcapst5_pan_epoch_20.hdf5 \
        https://github.com/anhvt00/MCAPS/raw/master/checkpoint/Pan/mcapst5_pan_epoch_20.hdf5 && \
    wget -q --show-progress -O ml_models/MCAPST5/checkpoints/xgboost_pan_epoch_20.bin \
        https://github.com/anhvt00/MCAPS/raw/master/checkpoint/Pan/xgboost_pan_epoch_20.bin && \
    wget -q --show-progress -O ml_models/MCAPST5/checkpoints/secstruct_checkpoint.pt \
        http://data.bioembeddings.com/public/embeddings/feature_models/t5/secstruct_checkpoint.pt && \
    echo "Model checkpoints downloaded successfully!"

# Pre-download HuggingFace ProtT5 model (optional but recommended for faster startup)
# This downloads ~3GB model. Uncomment if you want faster startup at cost of larger image.
# RUN python -c "from transformers import T5EncoderModel, T5Tokenizer; \
#     T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_half_uniref50-enc'); \
#     T5EncoderModel.from_pretrained('Rostlab/prot_t5_xl_half_uniref50-enc')"

# Copy application code
COPY . .

# Create entrypoint script
RUN echo '#!/bin/bash\n\
set -e\n\
\n\
echo "=== PPI Predictor API ===" \n\
echo "Checking environment..." \n\
\n\
# Check if required env vars are set\n\
if [ -z "$DB_CONNECTION_STRING" ] && [ -z "$DATABASE_URL" ] && [ -z "$SQLALCHEMY_DATABASE_URI" ]; then\n\
    echo "WARNING: No database connection string found. Using SQLite fallback."\n\
    export DB_CONNECTION_STRING="sqlite:///ppi_predictor.db"\n\
fi\n\
\n\
if [ -z "$JWT_SECRET" ]; then\n\
    echo "WARNING: JWT_SECRET not set. Using default (not secure for production!)"\n\
    export JWT_SECRET="dev-secret-change-me"\n\
fi\n\
\n\
if [ -z "$SECRET_KEY" ]; then\n\
    echo "WARNING: SECRET_KEY not set. Using default (not secure for production!)"\n\
    export SECRET_KEY="dev-secret-change-me"\n\
fi\n\
\n\
echo "Starting Flask application..."\n\
exec python app.py\n\
' > /app/entrypoint.sh && chmod +x /app/entrypoint.sh

# Expose port (development: 3000, production: 80)
EXPOSE 3000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:3000/ || exit 1

# Run entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]
