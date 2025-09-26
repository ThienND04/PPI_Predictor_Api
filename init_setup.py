import os
import subprocess

ROOT = "ml_models/MCAPST5"

# Tạo cấu trúc thư mục
os.makedirs(f"{ROOT}/checkpoints", exist_ok=True)
os.makedirs(f"{ROOT}/protT5_checkpoint", exist_ok=True)
os.makedirs(f"{ROOT}/output", exist_ok=True)

# Cài thư viện
subprocess.run('pip install transformers==4.29.2 sentencepiece==0.1.99 h5py==3.8.0', shell=True)
subprocess.run('pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html', shell=True)
subprocess.run('pip install xgboost matplotlib scikit-learn pandas gdown tensorflow==2.12.0 tensorflow-addons==0.20.0 pydot', shell=True)
subprocess.run('apt-get install -y graphviz wget', shell=True)

# Tải file mô hình & dữ liệu 
if not os.path.isfile(f"{ROOT}/checkpoints/mcapst5_pan_epoch_20.hdf5"):
    subprocess.run(f"wget -O {ROOT}/checkpoints/mcapst5_pan_epoch_20.hdf5 https://github.com/anhvt00/MCAPS/raw/master/checkpoint/Pan/mcapst5_pan_epoch_20.hdf5", shell=True)

if not os.path.isfile(f"{ROOT}/checkpoints/xgboost_pan_epoch_20.bin"):
    subprocess.run(f"wget -O {ROOT}/checkpoints/xgboost_pan_epoch_20.bin https://github.com/anhvt00/MCAPS/raw/master/checkpoint/Pan/xgboost_pan_epoch_20.bin", shell=True)

if not os.path.isfile(f"{ROOT}/checkpoints/secstruct_checkpoint.pt"):
    subprocess.run(f"wget -O {ROOT}/checkpoints/secstruct_checkpoint.pt http://data.bioembeddings.com/public/embeddings/feature_models/t5/secstruct_checkpoint.pt", shell=True)

if not os.path.isfile(f"{ROOT}/example_seqs.fasta"):
    subprocess.run(f"wget -O {ROOT}/example_seqs.fasta https://rostlab.org/~deepppi/example_seqs.fasta", shell=True)
