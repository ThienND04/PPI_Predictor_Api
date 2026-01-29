<div align="center">

# PPI Predictor API

### High-Performance Protein-Protein Interaction Prediction Service

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Flask](https://img.shields.io/badge/Flask-3.1.2-green.svg)](https://flask.palletsprojects.com/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](https://www.docker.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

*Powered by MCAPST5 & ProtT5 - State-of-the-art deep learning models for PPI prediction*

[Features](#-features) • [Quick Start](#-quick-start-with-docker) • [API Documentation](#-api-endpoints) • [Installation](#-manual-installation) • [Contributing](#-contributing)

</div>

---

## Overview

**PPI Predictor API** is a production-ready REST API service for predicting protein-protein interactions using the **MCAPST5** (Multi-Channel Attention Protein Structure T5) model. Built with modern ML frameworks and enterprise-grade architecture, it provides accurate PPI predictions through a simple HTTP interface.

### Technology Stack

| Category | Technologies |
|----------|-------------|
| **Backend** | Flask 3.1.2, SQLAlchemy 2.0, Flask-Migrate, Flask-Limiter |
| **Security** | JWT (PyJWT), Bcrypt, CORS |
| **ML/AI** | PyTorch 1.9.0, TensorFlow 2.12.0, Transformers 4.29.2 |
| **Models** | ProtT5 (Rostlab), MCAPST5, XGBoost |
| **Database** | PostgreSQL 15, SQLite (dev) |
| **DevOps** | Docker, Docker Compose |
| **Testing** | Pytest, Coverage |

---

## Quick Start with Docker (Recommended)

The fastest way to get PPI Predictor API running with all dependencies and models pre-configured.

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/) 20.10+
- [Docker Compose](https://docs.docker.com/compose/install/) 2.0+
- 8GB+ RAM (for model inference)
- 10GB+ disk space (for models and dependencies)

### Installation Steps

```bash
# 1. Clone the repository
git clone https://github.com/ThienND04/PPI_Predictor_Api.git
cd PPI_Predictor_Api

# 2. Create environment configuration
cp .env.example .env

# 3. Configure your secrets (edit .env file)
# Required: JWT_SECRET, SECRET_KEY
# Optional: Custom PostgreSQL credentials

# 4. Build and start services (first run: ~15-20 minutes)
docker-compose up --build

# 5. API is now live! 🎉
# Access at: http://localhost:3000
```

### Docker Management Commands

```bash
# Start services in background
docker-compose up -d

# View real-time logs
docker-compose logs -f api

# Stop all services
docker-compose down

# Rebuild after code changes
docker-compose up --build --force-recreate

# Start with PGAdmin (database GUI)
docker-compose --profile tools up
# PGAdmin: http://localhost:5050 (admin@admin.com / admin)

# Clean restart (removes all data)
docker-compose down -v && docker-compose up

# Check service health
docker-compose ps
```

### What Docker Does Automatically:
- ✅ Installs Python 3.9 with all dependencies
- ✅ Installs PyTorch & TensorFlow (CPU version)
- ✅ Downloads MCAPST5 model checkpoints (~500MB)
- ✅ Sets up PostgreSQL database
- ✅ Creates all database tables
- ✅ Caches HuggingFace ProtT5 model (~3GB, downloaded on first prediction)

---

## 📦 Manual Installation (Without Docker)

For development or custom deployment scenarios.

### System Requirements

- **Python**: 3.9 or higher
- **pip**: 21.0+
- **RAM**: 8GB+ recommended
- **Disk**: 10GB+ free space
- **OS**: Linux, macOS, or Windows with WSL2

### Installation Steps

#### 1. Clone Repository

```bash
git clone https://github.com/ThienND04/PPI_Predictor_Api.git
cd PPI_Predictor_Api
```

#### 2. Create Virtual Environment

```bash
# Linux/macOS
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

#### 3. Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install Python packages
pip install -r requirements.txt

# Install ML frameworks (may take 10-15 minutes)
pip install torch==1.9.0+cpu torchvision==0.10.0+cpu torchaudio==0.9.0 \
    -f https://download.pytorch.org/whl/torch_stable.html
pip install tensorflow==2.12.0 tensorflow-addons==0.20.0
```

#### 4. Download Model Checkpoints

```bash
# Create model directories
mkdir -p ml_models/MCAPST5/checkpoints

# Download MCAPST5 models (~500MB)
cd ml_models/MCAPST5/checkpoints

wget https://github.com/anhvt00/MCAPS/raw/master/checkpoint/Pan/mcapst5_pan_epoch_20.hdf5
wget https://github.com/anhvt00/MCAPS/raw/master/checkpoint/Pan/xgboost_pan_epoch_20.bin
wget http://data.bioembeddings.com/public/embeddings/feature_models/t5/secstruct_checkpoint.pt

cd ../../../
```

#### 5. Configure Environment

```bash
# Copy example environment file
cp .env.example .env

# Edit .env with your preferred editor
nano .env  # or vim, code, etc.
```

**Required Environment Variables:**

```env
# Application
FLASK_ENV=development
SECRET_KEY=your-secure-secret-key-here
JWT_SECRET=your-jwt-secret-key-here

# Database (choose one)
# SQLite (development)
SQLALCHEMY_DATABASE_URI=sqlite:///ppi_predictor.db

# PostgreSQL (production)
# DATABASE_URL=postgresql://user:password@localhost:5432/ppi_predictor
# or
# DB_CONNECTION_STRING=postgresql://user:password@localhost:5432/ppi_predictor

# Optional
FLASK_DEBUG=1
```

#### 6. Initialize Database

```bash
# Option A: Auto-initialization (recommended)
# Database and tables are created automatically on first run

# Option B: Manual migrations (advanced)
flask db init
flask db migrate -m "Initial migration"
flask db upgrade
```

#### 7. Run Application

```bash
# Development server
python app.py

# Alternative: Flask CLI
flask --app app run --debug --port 3000

# Production (use Gunicorn)
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:3000 "src.app_factory:create_app()"
```

```bash
# Activate virtual environment
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows
```

3) Cài đặt phụ thuộc
```bash
./init_setup.sh
# hoặc
pip install -r requirements.txt
```

4) Tạo file .env (ví dụ)
```
FLASK_ENV=development
SECRET_KEY=change-me
JWT_SECRET=change-me
SQLALCHEMY_DATABASE_URI=sqlite:///ppi.db
```

5) Khởi tạo CSDL (nếu dùng migrations)
```bash
flask db init
flask db migrate
flask db upgrade
```

6) Chạy ứng dụng
```bash
python app.py
# hoặc
flask --app app run --debug
```
