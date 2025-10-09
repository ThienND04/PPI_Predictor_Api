# PPI Predictor API

REST API for predicting protein–protein interactions using the MCAPST5 model.

- Tech: Flask, SQLAlchemy, Flask-Migrate, PyJWT, python-dotenv, pytest
- Status: Local development ready

## Prerequisites
- Python 3.9+
- pip and virtualenv

## Quick Start

1) Clone and enter project
```bash
git clone https://github.com/ThienND04/PPI_Predictor_Api.git
cd PPI_Predictor_Api
```

2) Create and activate virtual environment
```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
# venv\Scripts\activate   # Windows
```

3) Install dependencies
```bash
# If the script exists
./init_setup.sh
# or
pip install -r requirements.txt
```

4) Configure environment
Create a .env file in the project root:
```bash
cp .env.example .env  # if available, otherwise create manually
```
Example .env content:
```
FLASK_ENV=development
SECRET_KEY=change-me
JWT_SECRET=change-me
SQLALCHEMY_DATABASE_URI=sqlite:///ppi.db
# Alternatively: DATABASE_URL=postgresql+psycopg2://user:pass@localhost:5432/ppi
```

5) Initialize database (if using migrations)
```bash
flask db init     # first time only
flask db migrate  # generate migration
flask db upgrade  # apply migration
```

6) Run the app
```bash
python app.py
# or
flask --app app run --debug
```

## Testing
```bash
pytest -q
```

## Utilities
- List routes: `flask --app app routes`
- Format/lint (if configured): `ruff`, `black`, `isort`

## Notes
- Ensure model files (if required) are available and referenced via an env var (e.g., MODEL_PATH) or configured in the app.
- Default DB is SQLite via SQLALCHEMY_DATABASE_URI; switch to Postgres/MySQL by updating the URI.

---

## Hướng dẫn nhanh (VI)

1) Clone và vào thư mục dự án
```bash
git clone https://github.com/ThienND04/PPI_Predictor_Api.git
cd PPI_Predictor_Api
```

2) Tạo và kích hoạt môi trường ảo
```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
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
