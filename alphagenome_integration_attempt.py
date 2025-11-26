#step1
# AlphaGenome safe caller: chooses supported interval lengths, pads or windows sequences, runs predict_variant and returns stitched ref signal
from alphagenome.models import dna_client
from alphagenome.data import genome
import numpy as np, time, os

ALPHAGENOME_API_KEY = "ENTER_API_KEY"   # keep your key
model = dna_client.create(ALPHAGENOME_API_KEY)
print("AlphaGenome client created.")

# supported sizes reported by model (observed from error)
SUPPORTED_LENS = [16385, 131073, 524289, 1048577]

def choose_supported_len(L):
    """Return smallest supported length >= L (or the largest supported if L > max)."""
    for s in SUPPORTED_LENS:
        if L <= s:
            return s
    return SUPPORTED_LENS[-1]

def pad_sequence_to_len(seq, target_len):
    if len(seq) >= target_len:
        return seq[:target_len]
    # pad with Ns (harmless placeholder)
    return seq + ("N" * (target_len - len(seq)))

def window_ranges_for_sequence(L, window):
    """Return list of (start0, end0_exclusive) 0-based ranges to cover sequence by windows with 50% overlap."""
    if L <= window:
        return [(0, window)]
    stride = window // 2
    starts = list(range(0, max(1, L - window + 1), stride))
    if starts[-1] + window < L:
        starts.append(L - window)
    ranges = [(s, s + window) for s in starts]
    return ranges

def fetch_alphagenome_for_seq(seq_id, seq,
                              chrom='chrM',
                              ontology_terms=None,
                              outputs_list=None,
                              delay_between=0.2,
                              max_retries=4):
    """Fetch reference track for a sequence of any length. Returns 1D numpy array trimmed to original seq length."""
    ontology_terms = ontology_terms or ['UBERON:0000310']
    outputs_list = outputs_list or [dna_client.OutputType.RNA_SEQ]
    L = len(seq)
    window = choose_supported_len(L)
    # if seq shorter than window -> pad and run once
    if L <= window:
        padded = pad_sequence_to_len(seq, window)
        # use synthetic interval start positions (API indexes by positions; we just pick start=1)
        attempt = 0
        while attempt < max_retries:
            try:
                interval = genome.Interval(chromosome=chrom, start=1, end=window)
                dummy_var = genome.Variant(chromosome=chrom, position=1 + max(1, window//2),
                                           reference_bases='A', alternate_bases='C')
                outputs = model.predict_variant(interval=interval, variant=dummy_var,
                                                ontology_terms=ontology_terms,
                                                requested_outputs=outputs_list)
                # extract ref track if present
                if hasattr(outputs.reference, 'rna_seq') and outputs.reference.rna_seq is not None:
                    vals = np.asarray(outputs.reference.rna_seq.values, dtype=float)
                    return vals[:L]  # trim padding
                else:
                    return np.array([np.nan]*L)
            except Exception as e:
                attempt += 1
                wait = (2 ** attempt) + 0.2*attempt
                print(f"AlphaGenome call failed (attempt {attempt}): {e}. Backing off {wait:.1f}s")
                time.sleep(wait)
        raise RuntimeError("AlphaGenome failed after retries for padded sequence.")
    else:
        # seq longer than smallest supported: split into windows and stitch
        ranges = window_ranges_for_sequence(L, window)
        assembled = np.zeros( (len(ranges), window), dtype=float )
        got = np.zeros(len(ranges), dtype=bool)
        for i,(s0,e0) in enumerate(ranges):
            sub_seq = seq[s0:e0]
            padded = pad_sequence_to_len(sub_seq, window)  # should be equal length
            attempt = 0
            while attempt < max_retries:
                try:
                    interval = genome.Interval(chromosome=chrom, start=1, end=window)
                    dummy_var = genome.Variant(chromosome=chrom, position=1 + max(1, window//2),
                                               reference_bases='A', alternate_bases='C')
                    outputs = model.predict_variant(interval=interval, variant=dummy_var,
                                                    ontology_terms=ontology_terms,
                                                    requested_outputs=outputs_list)
                    if hasattr(outputs.reference, 'rna_seq') and outputs.reference.rna_seq is not None:
                        vals = np.asarray(outputs.reference.rna_seq.values, dtype=float)
                        assembled[i,:] = vals
                        got[i] = True
                    else:
                        assembled[i,:] = np.nan
                    break
                except Exception as e:
                    attempt += 1
                    wait = (2 ** attempt) + 0.2*attempt
                    print(f"AlphaGenome window call failed (attempt {attempt}): {e}. Backing off {wait:.1f}s")
                    time.sleep(wait)
            time.sleep(delay_between)
        # stitch by averaging overlaps
        final = np.full(L, np.nan, dtype=float)
        counts = np.zeros(L, dtype=int)
        for i,(s0,e0) in enumerate(ranges):
            seg = assembled[i,:]
            # map seg[0:window] onto final[s0:e0] (may overshoot if last window equals window)
            seg_len = min(window, e0 - s0)
            final[s0:s0+seg_len] = np.nan_to_num(final[s0:s0+seg_len]) + np.nan_to_num(seg[:seg_len])
            counts[s0:s0+seg_len] += 1
        # average where counts>0
        mask = counts > 0
        final[mask] = final[mask] / counts[mask]
        # positions with zero count -> nan
        return final

# Quick rate test using supported len (safe)
def quick_rate_test_safe(model, num_queries=2, seq_len=None):
    # if seq_len specified, pick nearest supported >= seq_len
    if seq_len is None:
        seq_len = SUPPORTED_LENS[0]
    s = choose_supported_len(seq_len)
    print("Running quick test with interval length:", s)
    t0 = time.time()
    successes = 0
    for i in range(num_queries):
        try:
            interval = genome.Interval(chromosome='chrM', start=1 + i*s, end=1 + i*s + s - 1)
            dummy_var = genome.Variant(chromosome='chrM', position=interval.start + max(1, s//2),
                                       reference_bases='A', alternate_bases='C')
            outputs = model.predict_variant(interval=interval, variant=dummy_var,
                                            ontology_terms=['UBERON:0000310'],
                                            requested_outputs=[dna_client.OutputType.RNA_SEQ])
            successes += 1
            print(f"Query {i+1}: success")
            time.sleep(0.1)
        except Exception as e:
            print(f"Query {i+1}: failed: {e}")
            break
    t1 = time.time()
    avg = (t1 - t0) / max(1, successes)
    print("Completed", successes, "successes in", (t1 - t0), "s -> avg", avg, "s/query")
    return avg

# Example: safe quick test using smallest supported length (16384)
avg_time = quick_rate_test_safe(model, num_queries=2, seq_len=16000)
print("Avg time per supported query (approx):", avg_time)

# Example usage for a real sequence object (seqs is your manifest list from earlier cells)
# For pilot:
# seq0 = seqs[0]['seq']
# ag_signal = fetch_alphagenome_for_seq(seqs[0]['id'], seq0)
# print("Signal length:", len(ag_signal))

#step2
import time
import numpy as np

AG_OUT = os.path.join(OUT_DIR, 'alphagenome')
os.makedirs(AG_OUT, exist_ok=True)

def fetch_alphagenome_tracks_for_sequence(seq_len, chrom='chrM', start_pos=1,
                                          ontology_terms=None, outputs_list=None,
                                          max_retries=4):
    """
    Returns the outputs object from model.predict_variant for the given interval.
    We'll use a dummy variant placed near the center because predict_variant expects a variant object.
    """
    ontology_terms = ontology_terms or ['UBERON:0000310']  # default tissue
    outputs_list = outputs_list or [dna_client.OutputType.RNA_SEQ, dna_client.OutputType.DNASE, dna_client.OutputType.CAGE]
    attempt = 0
    while attempt < max_retries:
        try:
            interval = genome.Interval(chromosome=chrom, start=start_pos, end=start_pos + seq_len - 1)
            dummy_var = genome.Variant(chromosome=chrom, position=start_pos + max(1, seq_len//2),
                                       reference_bases='A', alternate_bases='C')
            outputs = model.predict_variant(interval=interval, variant=dummy_var,
                                            ontology_terms=ontology_terms,
                                            requested_outputs=outputs_list)
            return outputs
        except Exception as e:
            attempt += 1
            # simple backoff
            wait = (2 ** attempt) + (0.2 * attempt)
            print(f"AlphaGenome call failed (attempt {attempt}/{max_retries}): {e}. Backing off {wait:.1f}s")
            time.sleep(wait)
    raise RuntimeError("AlphaGenome calls failed after retries.")

# Example wrapper: request RNA_SEQ values and save numpy arrays
def save_alphagenome_sequence(seq_id, seq_len, chrom='chrM', start_pos=1):
    outputs = fetch_alphagenome_tracks_for_sequence(seq_len, chrom=chrom, start_pos=start_pos)
    # extract reference.rna_seq values if present
    if hasattr(outputs.reference, 'rna_seq') and outputs.reference.rna_seq is not None:
        ref_vals = np.asarray(outputs.reference.rna_seq.values, dtype=float)
    else:
        ref_vals = None
    if hasattr(outputs.alternate, 'rna_seq') and outputs.alternate.rna_seq is not None:
        alt_vals = np.asarray(outputs.alternate.rna_seq.values, dtype=float)
    else:
        alt_vals = None
    savepath = os.path.join(AG_OUT, f"{seq_id}_alphagenome.npz")
    np.savez_compressed(savepath, ref=ref_vals, alt=alt_vals)
    print("Saved AlphaGenome arrays to", savepath)
    return savepath

# Pilot: test on first (short) genome to confirm
if len(seqs) > 0:
    s0 = seqs[0]
    try:
        save_alphagenome_sequence(s0['id'], len(s0['seq']), chrom='chrM', start_pos=1)
    except Exception as e:
        print("AlphaGenome test failed:", e)

#step3
# CELL: Convert AlphaGenome tracks -> peak intervals (BED-like) and summary CSV
# (This version ignores SegmentNT entirely and processes AlphaGenome .npz outputs)
#
# Requirements: AG_OUT should point to the folder where your alphagenome .npz files were saved.
# Each .npz is expected to contain an array named 'ref' (reference track). If 'alt' exists it's ignored here.
import os, glob, numpy as np, pandas as pd
from pathlib import Path

# Edit these paths if needed (they should match earlier cells)
AG_OUT = '/content/drive/MyDrive/mito_results/alphagenome'   # folder containing <seqid>_alphagenome.npz
OUT_DIR = '/content/drive/MyDrive/mito_results/processed'   # output folder for BEDs & summary
os.makedirs(OUT_DIR, exist_ok=True)

def track_to_intervals(values, pct=95, min_width=1):
    """
    Convert a 1D numpy array (values) into contiguous peak intervals using a percentile threshold.
    Returns list of (start, end_exclusive, peak_value).
    - start is 0-based inclusive
    - end_exclusive is Python-style end index (so BED-like end = end_exclusive)
    """
    if values is None:
        return []
    # ensure numeric and finite
    vals = np.asarray(values, dtype=float)
    if vals.size == 0:
        return []
    # replace inf with nan, and nan -> very small so threshold works
    vals = np.where(np.isfinite(vals), vals, np.nan)
    if np.all(np.isnan(vals)):
        return []
    thr = float(np.nanpercentile(vals, pct))
    mask = np.isfinite(vals) & (vals >= thr)
    intervals = []
    i = 0; L = len(vals)
    while i < L:
        if mask[i]:
            s = i
            while i < L and mask[i]:
                i += 1
            e = i  # e is exclusive
            if (e - s) >= min_width:
                peak_val = float(np.nanmax(vals[s:e]))
                intervals.append((int(s), int(e), peak_val))
        else:
            i += 1
    return intervals

# Process all .npz files in AG_OUT
npz_files = sorted(glob.glob(os.path.join(AG_OUT, '*_alphagenome.npz')))
summary_rows = []

for npz in npz_files:
    try:
        base = Path(npz).stem.replace('_alphagenome','')
        data = np.load(npz, allow_pickle=True)
        ref = data['ref'] if 'ref' in data.files else None
        # if ref is None but alt exists, use alt as best-effort
        if ref is None and 'alt' in data.files:
            ref = data['alt']
        if ref is None:
            print(f"Skipping {base}: no 'ref' or 'alt' array found in {npz}")
            continue
        # choose percentile (tune if needed)
        PCT = 95
        peaks = track_to_intervals(ref, pct=PCT, min_width=1)
        # write BED-like file (0-based start, end exclusive)
        bed_path = os.path.join(OUT_DIR, f"{base}_alphagenome_peaks.bed")
        with open(bed_path, 'w') as fh:
            for s,e,score in peaks:
                fh.write(f"chrM\t{s}\t{e}\t{score:.6f}\n")
        # gather summary
        total_peak_bp = sum((e - s) for s,e,_ in peaks)
        n_peaks = len(peaks)
        mean_peak_score = float(np.mean([p for _,_,p in peaks])) if n_peaks>0 else float('nan')
        summary_rows.append({
            'seq_id': base,
            'npz_path': npz,
            'bed_path': bed_path,
            'n_peaks': n_peaks,
            'total_peak_bp': int(total_peak_bp),
            'mean_peak_score': mean_peak_score,
            'pct_threshold': PCT,
            'signal_length': int(len(ref))
        })
        print(f"Processed {base}: {n_peaks} peaks, {total_peak_bp} bp -> {bed_path}")
    except Exception as exc:
        print(f"Error processing {npz}: {exc}")

# Save summary CSV
summary_df = pd.DataFrame(summary_rows)
summary_csv = os.path.join(OUT_DIR, "alphagenome_peak_summary.csv")
summary_df.to_csv(summary_csv, index=False)
print("Wrote summary:", summary_csv)
