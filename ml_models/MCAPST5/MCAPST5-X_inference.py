import subprocess
import os

import numpy as np
import torch
from transformers import T5EncoderModel, T5Tokenizer
import tensorflow as tf
from tensorflow.keras.models import load_model
import time
from tqdm import tqdm
import h5py

# Constants
MAX_SEQ_LEN = 1200
EMB_DIM = 1024

# Load ProtT5 encoder
model_encoder = None
tokenizer = None
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

# In the following you can define your desired output. Current options:
# per_residue embeddings
# per_protein embeddings
# secondary structure predictions

# Replace this file with your own (multi-)FASTA
# Headers are expected to start with ">";
# seq_path = "guo.fasta"

# whether to retrieve embeddings for each residue in a protein
# --> Lx1024 matrix per protein with L being the protein's length
# as a rule of thumb: 1k proteins require around 1GB RAM/disk
per_residue = True
# per_residue_path = "./output/per_residue_embeddings.h5" # where to store the embeddings

# whether to retrieve per-protein embeddings
# --> only one 1024-d vector per protein, irrespective of its length
per_protein = False
# per_protein_path = "./output/per_protein_embeddings.h5" # where to store the embeddings

# whether to retrieve secondary structure predictions
# This can be replaced by your method after being trained on ProtT5 embeddings
sec_struct = False
# sec_struct_path = "./output/ss3_preds.fasta" # file for storing predictions

# make sure that either per-residue or per-protein embeddings are stored
assert per_protein is True or per_residue is True or sec_struct is True, print(
    "Minimally, you need to active per_residue, per_protein or sec_struct. (or any combination)")

def get_T5_model():
    global model_encoder, tokenizer
    if model_encoder is None or tokenizer is None:
        model_encoder = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc").to(device).eval()
        tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc", do_lower_case=False)
    return model_encoder, tokenizer

# Embedding generator
def get_embeddings(model, tokenizer, seqs, max_residues=4000, max_seq_len=1000, max_batch=100):
    results = {"residue_embs": dict(),
               "protein_embs": dict(),
               "sec_structs": dict()
               }

    # sort sequences according to length (reduces unnecessary padding --> speeds up embedding)
    seq_dict   = sorted( seqs.items(), key=lambda kv: len( seqs[kv[0]] ), reverse=True )
    start = time.time()
    batch = list()
    for seq_idx, (pdb_id, seq) in tqdm(enumerate(seq_dict,1)):
        seq = seq
        seq_len = len(seq)
        seq = ' '.join(list(seq))
        batch.append((pdb_id, seq, seq_len))

        # count residues in current batch and add the last sequence length to
        # avoid that batches with (n_res_batch > max_residues) get processed
        n_res_batch = sum([s_len for _, _, s_len in batch]) + seq_len
        if len(batch) >= max_batch or n_res_batch >= max_residues or seq_idx == len(seq_dict) or seq_len > max_seq_len:
            pdb_ids, seqs, seq_lens = zip(*batch)
            batch = list()

            # add_special_tokens adds extra token at the end of each sequence
            token_encoding = tokenizer.batch_encode_plus(seqs, add_special_tokens=True, padding="longest")
            input_ids = torch.tensor(token_encoding['input_ids']).to(device)
            attention_mask = torch.tensor(token_encoding['attention_mask']).to(device)

            try:
                with torch.no_grad():
                    # returns: ( batch-size x max_seq_len_in_minibatch x embedding_dim )
                    embedding_repr = model(input_ids, attention_mask=attention_mask)
            except RuntimeError:
                print("RuntimeError during embedding for {} (L={})".format(pdb_id, seq_len))
                continue

            if sec_struct:  # in case you want to predict secondary structure from embeddings
                # d3_Yhat, d8_Yhat, diso_Yhat = sec_struct_model(embedding_repr.last_hidden_state)
                pass

            for batch_idx, identifier in enumerate(pdb_ids):  # for each protein in the current mini-batch
                s_len = seq_lens[batch_idx]
                # slice off padding --> batch-size x seq_len x embedding_dim
                emb = embedding_repr.last_hidden_state[batch_idx, :s_len]
                if sec_struct:  # get classification results
                    # results["sec_structs"][identifier] = torch.max(d3_Yhat[batch_idx, :s_len], dim=1)[1].detach().cpu().numpy().squeeze()
                    pass
                if per_residue:  # store per-residue embeddings (Lx1024)
                    results["residue_embs"][identifier] = emb.detach().cpu().numpy().squeeze()
                if per_protein:  # apply average-pooling to derive per-protein embeddings (1024-d)
                    protein_emb = emb.mean(dim=0)
                    results["protein_embs"][identifier] = protein_emb.detach().cpu().numpy().squeeze()

    passed_time = time.time() - start
    avg_time = passed_time / len(results["residue_embs"]) if per_residue else passed_time / len(results["protein_embs"])
    print('\n############# EMBEDDING STATS #############')
    print('Total number of per-residue embeddings: {}'.format(len(results["residue_embs"])))
    print('Total number of per-protein embeddings: {}'.format(len(results["protein_embs"])))
    print("Time for generating embeddings: {:.1f}[m] ({:.3f}[s/protein])".format(
        passed_time / 60, avg_time))
    print('\n############# END #############')
    return results

#Write embedding to disk
def save_embeddings(emb_dict,out_path):
    with h5py.File(str(out_path), "w") as hf:
        for sequence_id, embedding in emb_dict.items():
            # noinspection PyUnboundLocalVariable
            hf.create_dataset(sequence_id, data=embedding)
    return None

# Padding function
def pad(rst, length=MAX_SEQ_LEN, dim=EMB_DIM):
    if len(rst) > length:
        return rst[:length]
    elif len(rst) < length:
        return np.concatenate((rst, np.zeros((length - len(rst), dim))))
    return rst

# Load main classifier model
classifier = None
def load_classifier(model_path="ml_models/MCAPST5/checkpoints/mcapst5_pan_epoch_20.hdf5"):
    global classifier
    if classifier is None:
        classifier = load_model(model_path)
    return classifier

def predict_interaction(protein1_id: str, protein1_seq: str, protein2_id: str, protein2_seq: str) -> float:
    # Nhận vào 2 chuỗi protein và trả về xác suất tương tác của chúng.

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
    print(f"{score: .4f}")