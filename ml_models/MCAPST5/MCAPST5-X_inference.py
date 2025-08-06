import subprocess
import os

import numpy as np
import torch
from transformers import T5EncoderModel, T5Tokenizer
import tensorflow as tf
from tensorflow.keras.models import load_model

# Constants
MAX_SEQ_LEN = 1200
EMB_DIM = 1024

# Load ProtT5 encoder
model_encoder = None
tokenizer = None
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

def get_T5_model():
    global model_encoder, tokenizer
    if model_encoder is None or tokenizer is None:
        model_encoder = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc").to(device).eval()
        tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc", do_lower_case=False)
    return model_encoder, tokenizer

# Embedding generator
def get_embeddings(model, tokenizer, seqs, max_residues=2400, max_seq_len=1200, max_batch=2):
    results = {"residue_embs": dict()}
    seq_dict = sorted(seqs.items(), key=lambda kv: len(kv[1]), reverse=True)
    batch = []
    for seq_idx, (pid, seq) in enumerate(seq_dict):
        seq_len = len(seq)
        seq = ' '.join(list(seq))
        batch.append((pid, seq, seq_len))

        n_res_batch = sum([s_len for _, _, s_len in batch]) + seq_len
        if len(batch) >= max_batch or n_res_batch >= max_residues or seq_idx == len(seq_dict)-1:
            pdb_ids, seqs_batch, seq_lens = zip(*batch)
            batch = []

            token_encoding = tokenizer.batch_encode_plus(seqs_batch, add_special_tokens=True, padding="longest")
            input_ids = torch.tensor(token_encoding['input_ids']).to(device)
            attention_mask = torch.tensor(token_encoding['attention_mask']).to(device)

            with torch.no_grad():
                embedding_repr = model(input_ids, attention_mask=attention_mask)

            for i, identifier in enumerate(pdb_ids):
                s_len = seq_lens[i]
                emb = embedding_repr.last_hidden_state[i, :s_len]
                results["residue_embs"][identifier] = emb.detach().cpu().numpy()
    return results

# Padding function
def pad(seq_emb, length=MAX_SEQ_LEN, dim=EMB_DIM):
    if len(seq_emb) > length:
        return seq_emb[:length]
    elif len(seq_emb) < length:
        return np.concatenate((seq_emb, np.zeros((length - len(seq_emb), dim))))
    return seq_emb

# Load main classifier model
classifier = None
def load_classifier(model_path="ml_models/MCAPST5/checkpoints/mcapst5_pan_epoch_20.hdf5"):
    global classifier
    if classifier is None:
        classifier = load_model(model_path)
    return classifier

def predict_interaction(protein1_id: str, protein1_seq: str, protein2_id: str, protein2_seq: str) -> float:
    """
    Nhận vào 2 chuỗi protein và trả về xác suất tương tác của chúng.
    """
    model, tokenizer = get_T5_model()
    clf = load_classifier()

    protein1_seq = protein1_seq[:MAX_SEQ_LEN]
    protein2_seq = protein2_seq[:MAX_SEQ_LEN]

    seqs = {
        protein1_id: protein1_seq,
        protein2_id: protein2_seq,
    }

    embeddings = get_embeddings(model, tokenizer, seqs)

    x1 = pad(embeddings["residue_embs"][protein1_id])
    x2 = pad(embeddings["residue_embs"][protein2_id])

    x1 = np.expand_dims(x1, axis=0)
    x2 = np.expand_dims(x2, axis=0)

    pred = clf.predict([x1, x2])[0][0]
    return float(pred)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Dự đoán tương tác giữa 2 protein")
    parser.add_argument("--id1", required=True, help="ID protein 1")
    parser.add_argument("--seq1", required=True, help="Chuỗi protein 1")
    parser.add_argument("--id2", required=True, help="ID protein 2")
    parser.add_argument("--seq2", required=True, help="Chuỗi protein 2")
    args = parser.parse_args()

    score = predict_interaction(args.id1, args.seq1, args.id2, args.seq2)
    print(f"Khả năng tương tác: {score:.4f}")