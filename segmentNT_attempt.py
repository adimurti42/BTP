#step1
from google.colab import drive
drive.mount('/content/drive')

#step2
!pip install --upgrade pip
!pip install transformers biopython numpy pandas matplotlib seaborn tqdm
!git clone -q https://huggingface.co/InstaDeepAI/segment_nt segment_nt_repo || true

#step3
# Assumes Drive is already mounted at /content/drive
import os, glob, json, time, math
from pathlib import Path
from tqdm import tqdm
from Bio import SeqIO
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Set your paths (edit if you used other locations)
FASTA_DIR = '/content/drive/MyDrive/FASTA'   # <- put your fasta files here
OUT_DIR   = '/content/drive/MyDrive/Output'  # <- outputs will be saved here
os.makedirs(OUT_DIR, exist_ok=True)

# Quick listing
fpaths = sorted(glob.glob(os.path.join(FASTA_DIR, '*.*')))
print(f"Found {len(fpaths)} files under {FASTA_DIR}")

#step4
# This reads all sequences in all FASTA files and builds a list of dicts: {id, seq, source_file}
def load_all_sequences(fasta_dir):
    seqs = []
    from Bio import SeqIO
    supported = ('*.fa','*.fasta','*.fna','*.fa.gz','*.fasta.gz')
    files = []
    for pat in supported:
        files += glob.glob(os.path.join(fasta_dir, pat))
    files = sorted(set(files))
    for fp in files:
        for rec in SeqIO.parse(fp, 'fasta'):
            seq = str(rec.seq).upper()
            # optional cleanup: remove ambiguous N if you wish
            # seq = seq.replace('N','')
            seqs.append({'id': rec.id, 'seq': seq, 'file': os.path.basename(fp)})
    return seqs

seqs = load_all_sequences(FASTA_DIR)
print("Loaded", len(seqs), "sequences (genomes). Example IDs:", [s['id'] for s in seqs[:5]])
# Save small manifest
pd.DataFrame(seqs)[:10]

#step5
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Example with BERT
model_name = "bert-base-uncased" # Or "roberta-base", "distilbert-base-uncased", etc.

tokenizer = AutoTokenizer.from_pretrained(model_name)
# num_labels should be set according to the number of classes in your specific token classification task
# For this example, we'll set it to 2 (a common placeholder for binary classification)
model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=2)
model.to(device)
model.eval() # Set model to evaluation mode

print(f"Loaded {model_name} for token classification on device: {device}")

# You can now use 'tokenizer' and 'model' to process your sequences for token classification.

#step6
from huggingface_hub import login
login(token="YOUR_TOKEN")   # <- replace with your token (string)

from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_id = "InstaDeepAI/segment_nt"   # mitochondria model you selected

# Retry load with trust_remote_code (now authenticated)
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForTokenClassification.from_pretrained(model_id, trust_remote_code=True).to(device)
model.eval()
print("Loaded SegmentNT model:", model_id, "on device:", device)

!git clone https://github.com/InstaDeepAI/nucleotide-transformer.git instadeep_nt_repo

import sys, os
# Corrected: repo_path should be the directory where the repo was cloned
repo_path = '/content/segment_nt_repo'   # path after clone

# Ensure the repo_path is in sys.path so modules can be imported
if repo_path not in sys.path:
    sys.path.insert(0, repo_path)

# Inspect repo README or inference script; recommended file: inference/inference_segment_nt.py or similar
# We'll try to import their inference utility (the exact module name may vary in the repo)
# Try the commonly used module name from repo:
try:
    from modeling_segment_nt import SegmentNTInference  # Try importing from modeling_segment_nt.py
    print("Imported SegmentNTInference from repo")
except Exception as e:
    print("Could not import inference helper directly:", e)
    print("Listing files in repo root to find the right script:")
    # List contents of the directory, not the .ipynb file
    print(os.listdir(repo_path))
  
#step7
import importlib.util, sys, os

!pip install haiku

# Corrected file_path to only include the file path, not line numbers or class def.
file_path = "/content/nucleotide-transformer/nucleotide_transformer/model.py"
print("Loading module from:", file_path)
spec = importlib.util.spec_from_file_location("segment_module", file_path)

# Check if spec is None, which indicates the file was not found or could not be loaded
if spec is None:
    raise FileNotFoundError(f"Could not find or load module from: {file_path}. Please verify the path.")

segment_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(segment_module)

# Find candidate class names
candidates = [name for name in dir(segment_module) if not name.startswith("_")]
print("Exported names from module:", candidates)

# Try to get SegmentNTConfig, as implied by the original file_path string
# If a different class is intended (e.g., SegmentNTModel), adjust this line.
SegmentNTConfig_Class = getattr(segment_module, "SegmentNTConfig", None)
if SegmentNTConfig_Class is None:
    raise RuntimeError("SegmentNTConfig class not found in the module. Check the exported names above.")
print("Found SegmentNTConfig class:", SegmentNTConfig_Class)

#step8
# The model's inference tokenization behavior can vary; InstaDeep's inference notebook shows a safe pattern:
# 1) Slide a window across your sequence (window_size in bases)
# 2) Tokenize window and run through model
# 3) Map token outputs back to base-resolution (tokenizer may use k-mer tokens; use the tokenizer's offset mapping)

# We'll implement a conservative sliding-window approach with overlaps, then stitch outputs by majority at overlapping positions.

def segmentnt_predict_sequence(sequence, window_bases=30000, stride_bases=15000):
    """
    Predict per-base class labels for `sequence` using sliding windows.
    Returns an integer numpy array of length == len(sequence) with predicted class id per base.
    """
    L = len(sequence)
    out_labels = np.full(L, fill_value=-1, dtype=int)  # -1 = unknown
    counts = np.zeros(L, dtype=int)

    # sliding windows
    starts = list(range(0, max(1, L - window_bases + 1), stride_bases))
    if starts[-1] + window_bases < L:
        starts.append(L - window_bases)
    if len(starts) == 0:
        starts = [0]

    for s in starts:
        e = min(L, s + window_bases)
        subseq = sequence[s:e]
        # Tokenize. Use return_offsets_mapping to map tokens back to base positions if tokenizer supports it.
        inputs = tokenizer(subseq, return_tensors='pt', truncation=True, max_length=window_bases, return_offsets_mapping=True)
        # Move to device
        inputs = {k: v.to(device) for k, v in inputs.items() if k != 'offset_mapping'}
        offsets = tokenizer(subseq, return_offsets_mapping=True)['offset_mapping']  # list of tuples / pairs
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits[0].cpu().numpy()  # (token_len, num_classes)
        preds = logits.argmax(axis=-1)  # (token_len,)
        # Map token predictions to base positions using offsets (offsets in characters within subseq)
        # offsets is a list of (start, end) for each token in characters (bases)
        # If offsets is empty / tokenizer doesn't support, fallback to naive per-base mapping
        if len(offsets) == len(preds):
            for tok_idx, (tstart, tend) in enumerate(offsets):
                # tstart,tend are positions inside subseq where this token spans
                if tstart is None or tend is None:
                    continue
                if tstart == tend:
                    continue
                base_global_start = s + tstart
                base_global_end = s + tend  # exclusive
                # assign prediction to bases in this token span
                out_labels[base_global_start:base_global_end] += preds[tok_idx] + 1  # accumulate
                counts[base_global_start:base_global_end] += 1
        else:
            # fallback: if tokenizer returns 1 token per base (rare) or mismatch, expand tokens uniformly:
            tk_len = len(preds)
            # map tokens to equal-sized chunks
            approx_chunk = max(1, (e - s) // tk_len)
            pos = s
            for tok_idx in range(tk_len):
                start_pos = pos
                end_pos = min(e, pos + approx_chunk)
                out_labels[start_pos:end_pos] += preds[tok_idx] + 1
                counts[start_pos:end_pos] += 1
                pos = end_pos

    # Finalize by majority vote where counts>0
    final = np.full(L, fill_value=-1, dtype=int)
    mask = counts > 0
    final[mask] = (out_labels[mask] / counts[mask]).round().astype(int)
    # For any positions still -1, set to background class 0
    final[final < 0] = 0
    return final

# Small test (run on first sequence, but keep subseq short to debug)
if len(seqs) > 0:
    test_seq = seqs[0]['seq'][:1000]
    lab = segmentnt_predict_sequence(test_seq, window_bases=1024, stride_bases=512)
    print("Test labels length:", len(lab), "unique classes:", np.unique(lab))
