import os
from pathlib import Path 

PROJECT_NAME = "Protein-Protein Interaction Prediction" #add your project name here

LIST_FILES = [
    "Dockerfile",
    ".env",
    ".gitignore",
    "app.py",
    "init_setup.py",
    "README.md",
    "requirements.txt",
    "src/__init__.py",
    # config folder
    "src/config/__init__.py",
    "src/config/config.py",
    "src/config/dev_config.py",
    "src/config/production_config.py",
    # api
    "src/api/__init__.py",
    # middlewares
    "src/middlewares/__init__.py",
    # schemas
    "src/schemas/__init__.py",
    "src/schemas/Protein.py",
    # services
    "src/services/__init__.py",
    "src/services/jwt_service.py",
    # api and utils
     "src/api.py",
     "src/utils.py",
   ]

for file_path in LIST_FILES:
    file_path = Path(file_path)
    file_dir, file_name = os.path.split(file_path)

    # first make dir
    if file_dir!="":
        os.makedirs(file_dir, exist_ok= True)
        print(f"Creating Directory: {file_dir} for file: {file_name}")
    
    if (not os.path.exists(file_path)) or (os.path.getsize(file_path)==0):
        with open(file_path, "w") as f:
            pass
            print(f"Creating an empty file: {file_path}")
    else:
        print(f"File already exists {file_path}")