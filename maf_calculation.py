import os
import sys
import re
import json
import argparse
from typing import List, Dict, Any
import pandas as pd
import numpy as np
from time import sleep

# Import AlphaGenome client
try:
    from alphagenome.models import dna_client
    from alphagenome.data import genome
except ImportError:
    dna_client = None
    genome = None

# ---------- helpers ----------
def safe_float(x):
    try:
        if pd.isna(x): return np.nan
        s = str(x).strip().replace(',', '')
        if s in ['', '.', 'nan', 'NA', 'None']: return np.nan
        v = float(s)
        return v
    except Exception:
        return np.nan

def compute_maf_from_af(af):
    if pd.isna(af): return np.nan
    if af > 1:
        if af <= 100:
            af = af / 100.0
        else:
            return np.nan
    af = max(0.0, min(1.0, af))
    return min(af, 1.0 - af)

def find_af_columns(df: pd.DataFrame) -> List[str]:
    candidates = []
    for c in df.columns:
        low = c.lower()
        if '_af' in low or low.endswith('af') or 'gnomad' in low or 'regeneron' in low or 'exome_af' in low:
            candidates.append(c)
        elif re.search(r'\baf\b', c, re.I) and c not in ['AVAIL', 'AFTER']:
            candidates.append(c)
    return sorted(set(candidates))

def construct_variant_id(row, chrom_col, pos_col, ref_col, alt_col):
    try:
        return f"{str(row[chrom_col]).strip()}:{str(row[pos_col]).strip()}:{str(row[ref_col]).strip()}:{str(row[alt_col]).strip()}"
    except Exception:
        return None

# ---------- AlphaGenome client interaction ----------

def alphagenome_batch_predict(variants: List[Dict[str,Any]], api_key: str, batch_size: int=10, sleep_between: float=1.0) -> List[Dict[str,Any]]:
    """
    Use AlphaGenome official Python client to predict variant effects.
    - variants: list of dicts with 'chrom', 'pos', 'ref', 'alt', 'id'
    - api_key: your AlphaGenome API key
    - Returns: list of predictions compatible with old API (id, score, label, details)
    """
    if dna_client is None or genome is None:
        raise RuntimeError("alphagenome package is not installed. Please run: pip install alphagenome")

    try:
        model = dna_client.create(api_key)
    except Exception as e:
        raise RuntimeError(f"Failed to create AlphaGenome client: {e}")

    # Import protobuf converter
    from google.protobuf.json_format import MessageToDict

    all_preds = []
    processed = 0

    for i in range(0, len(variants), batch_size):
        batch = variants[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1}, variants {i+1}-{min(i+batch_size, len(variants))}")
        
        for var in batch:
            try:
                chrom = var.get('chrom')
                pos = int(var.get('pos')) if 'pos' in var and var.get('pos') is not None else None
                ref = var.get('ref')
                alt = var.get('alt')
                var_id = var.get('id', None)

                if chrom and pos and ref and alt:
                    # Keep 'chr' prefix for AlphaGenome API
                    chrom_formatted = chrom if chrom.startswith('chr') else f'chr{chrom}'
                    
                    variant_obj = genome.Variant(
                        chromosome=chrom_formatted, 
                        position=pos, 
                        reference_bases=ref, 
                        alternate_bases=alt
                    )
                    
                    # Use 2048bp window (minimum supported size)
                    window_size = 1024
                    interval = genome.Interval(
                        chromosome=chrom_formatted, 
                        start=max(1, pos-window_size), 
                        end=pos+window_size
                    )

                    # Make prediction
                    outputs = model.predict_variant(
                        interval=interval, 
                        variant=variant_obj, 
                        requested_outputs=[dna_client.OutputType.RNA_SEQ],
                        ontology_terms=[]
                    )

                    # FIXED: Extract predictions from protobuf object
                    score = None
                    label = "predicted"
                    details = {}
                    
                    if outputs:
                        try:
                            outputs_dict = MessageToDict(outputs)
                            for key, val in outputs_dict.items():
                                details[str(key)] = str(val)[:100]  # Truncate long outputs
                                
                                # Try to extract numerical scores
                                if "score" in str(key).lower() or "effect" in str(key).lower():
                                    try:
                                        if isinstance(val, (int, float)):
                                            score = float(val)
                                        elif isinstance(val, list) and len(val) > 0:
                                            score = float(val[0]) if isinstance(val[0], (int, float)) else None
                                    except:
                                        pass
                        except Exception as e:
                            details["parsing_error"] = str(e)[:50]
                    
                    pred = {
                        "id": var_id, 
                        "score": score, 
                        "label": label, 
                        "details": details
                    }
                    all_preds.append(pred)
                    processed += 1
                    print(f"    â Successfully predicted {var_id}")
                    
                else:
                    all_preds.append({
                        "id": var_id, 
                        "score": None, 
                        "label": "missing_coordinates", 
                        "details": {"error": "Missing chromosome/position/ref/alt"}
                    })
                    processed += 1
                    
            except Exception as e:
                error_msg = str(e)[:200]
                all_preds.append({
                    "id": var.get('id', None), 
                    "score": None, 
                    "label": "prediction_error", 
                    "details": {"error": error_msg}
                })
                processed += 1
                print(f"  Warning: Failed to predict variant {var.get('id')}: {error_msg}")

        print(f"  Completed {processed}/{len(variants)} variants")
        if i + batch_size < len(variants):
            sleep(sleep_between)

    return all_preds

# ---------- main pipeline ----------
def main(args):
    api_key = args.api_key or os.environ.get("ALPHAGENOME_API_KEY")

    if args.do_alphagenome and not api_key:
        print("AlphaGenome API key required for predictions. Provide via --api-key or ALPHAGENOME_API_KEY environment variable.")
        sys.exit(1)

    df = pd.read_csv(args.input, dtype=str)
    # Normalize columns by stripping whitespace
    df.rename(columns={c:c.strip() for c in df.columns}, inplace=True)

    # Find core columns
    chrom_col = next((c for c in df.columns if c.lower() == 'chrom'), None)
    pos_col   = next((c for c in df.columns if c.lower() == 'pos'), None)
    ref_col   = next((c for c in df.columns if c.lower() == 'ref'), None)
    alt_col   = next((c for c in df.columns if c.lower() == 'alt'), None)
    rsid_col  = next((c for c in df.columns if c.lower() in ('avsnp151','rsid','rs_id')), None)

    # Build variant_id
    if chrom_col and pos_col and ref_col and alt_col:
        df['variant_id'] = df.apply(lambda r: construct_variant_id(r, chrom_col, pos_col, ref_col, alt_col), axis=1)
    elif rsid_col:
        df['variant_id'] = df[rsid_col].fillna('').astype(str)
    else:
        df['variant_id'] = df.index.astype(str)

    # Detect AF columns and compute MAFs
    af_cols = find_af_columns(df)
    print("AF-like columns detected:", af_cols)
    for c in af_cols:
        safe_vals = df[c].map(safe_float)
        df['MAF__' + re.sub(r'[^0-9A-Za-z_]', '_', c)] = safe_vals.map(compute_maf_from_af)

    # Compute best available MAF using priority heuristics (IGVC-like â gnomad â others)
    priority = []
    for p in ['igvc','igv','indigen','gnomad','regeneron','exome_af','af']:
        for c in af_cols:
            if p in c.lower() and c not in priority:
                priority.append(c)
    for c in af_cols:
        if c not in priority:
            priority.append(c)
    def pick_best_maf(row):
        for c in priority:
            val = row.get('MAF__' + re.sub(r'[^0-9A-Za-z_]', '_', c), np.nan)
            if not pd.isna(val):
                return val, c
        return np.nan, None
    bests = df.apply(pick_best_maf, axis=1, result_type='expand')
    df['MAF_best'] = bests[0]
    df['MAF_best_from'] = bests[1]

    # Prepare variants payload for AlphaGenome
    variants_for_api = []
    for _, row in df.iterrows():
        # only include variants with coordinates; fallback to rsid if no coords
        if chrom_col and pos_col and ref_col and alt_col and pd.notna(row['variant_id']):
            # parse variant_id into fields if already in chr:pos:ref:alt
            parts = str(row['variant_id']).split(':')
            if len(parts) >= 4:
                chrom, pos, ref, alt = parts[0], parts[1], parts[2], parts[3]
            else:
                chrom, pos, ref, alt = None, None, None, None
            variant_obj = {"id": row['variant_id']}
            if chrom: variant_obj.update({"chrom": chrom})
            if pos:
                try: variant_obj["pos"] = int(pos)
                except: variant_obj["pos"] = pos
            if ref: variant_obj["ref"] = ref
            if alt: variant_obj["alt"] = alt
            # optionally include rsid if present
            if rsid_col and pd.notna(row.get(rsid_col, None)):
                variant_obj["rsid"] = str(row[rsid_col])
            variants_for_api.append(variant_obj)
        else:
            # fallback include the variant_id only
            variants_for_api.append({"id": row['variant_id']})

    # Call AlphaGenome (batch)
    preds = []
    if args.do_alphagenome:
        print(f"Sending {len(variants_for_api)} variants to AlphaGenome API in batches...")
        preds = alphagenome_batch_predict(variants_for_api, api_key, batch_size=args.batch_size, sleep_between=args.sleep_between)
        print("Received", len(preds), "predictions (raw).")
        # normalize preds: expect list of dicts with at least 'id' and 'score' or 'prediction' keys
    else:
        print("Skipping AlphaGenome API call (--no-alphagenome was set).")

    # Convert predictions to dataframe & merge
    if preds:
        # adapt based on returned structure; we try to handle a few likely patterns
        # expected minimal: [{'id':'chr17:...', 'score':0.82, 'label':'...'}, ...]
        pred_rows = []
        for p in preds:
            # try common fields
            pid = p.get('id') or p.get('variant_id') or p.get('name') or p.get('key')
            score = p.get('score') if 'score' in p else p.get('prediction_score') or p.get('alpha_score')
            label = p.get('label') or p.get('prediction') or p.get('call')
            details = {k:v for k,v in p.items() if k not in ('id','variant_id','name','key','score','prediction_score','alpha_score','label','prediction','call')}
            pred_rows.append({"variant_id": pid, "alphagenome_score": score, "alphagenome_label": label, "alphagenome_details": json.dumps(details)})
        pred_df = pd.DataFrame(pred_rows)
        # merge preds into main df on variant_id
        df = df.merge(pred_df, on='variant_id', how='left')
    else:
        # if no preds, create empty columns so output has consistent schema
        df['alphagenome_score'] = np.nan
        df['alphagenome_label'] = None
        df['alphagenome_details'] = None

    # Save summary
    out_prefix = os.path.splitext(os.path.basename(args.input))[0]
    out_csv = out_prefix + '_alpha_maf_summary.csv'
    # choose a sensible set of columns to keep (variant_id, rsid, gene, consequence, quality, MAFs, alpha columns)
    keep_cols = []
    for want in ['variant_id','avsnp151','Gene.refGeneWithVer','Func.refGeneWithVer','ExonicFunc.refGeneWithVer',
                 'MC','cDNA change','AA Change','AAChange.refGeneWithVer','QUAL','DP','CADD_phred','CLNSIG']:
        if want in df.columns:
            keep_cols.append(want)
    maf_cols = [c for c in df.columns if c.startswith('MAF__')]
    keep_cols += maf_cols + ['MAF_best','MAF_best_from','alphagenome_score','alphagenome_label','alphagenome_details']
    # ensure unique
    keep_cols = [c for c in keep_cols if c in df.columns]
    summary = df[keep_cols].copy()
    summary.to_csv(out_csv, index=False)
    print("Wrote summary to:", out_csv)

    # Prepare IGVC query file: prefer rsIDs, else variant_id
    qlist_file = out_prefix + '_igvc_queries.txt'
    with open(qlist_file, 'w') as fh:
        for _, row in df.iterrows():
            q = None
            if rsid_col and pd.notna(row.get(rsid_col, None)) and str(row.get(rsid_col)).strip() not in ['.','NA','nan','']:
                q = str(row.get(rsid_col)).strip()
            else:
                q = str(row.get('variant_id'))
            fh.write(q + '\n')
    print("Wrote IGVC query list to:", qlist_file)

    print("Pipeline complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AlphaGenome + MAF pipeline")
    parser.add_argument("--input", "-i", required=True, help="Input variants CSV file")
    parser.add_argument("--api-key", help="AlphaGenome API key (optional; can use ALPHAGENOME_API_KEY env var)")
    parser.add_argument("--no-alphagenome", dest="do_alphagenome", action="store_false", help="Skip calling AlphaGenome (only compute MAF and make query list)")
    parser.add_argument("--batch-size", type=int, default=20, help="Batch size for AlphaGenome API calls")
    parser.add_argument("--sleep-between", type=float, default=1.0, help="Seconds to sleep between batches")
    args = parser.parse_args()
    # by default do_alphagenome True unless flag set
    if getattr(args, "do_alphagenome", True) is None:
        args.do_alphagenome = True
    main(args)
