"""
eval_ensemble.py — Standalone evaluation script for GTE-PPIS-baseline.

Usage:
    python eval_ensemble.py [--log_dir LOG_DIR] [--out_dir OUT_DIR] [--npy_file NPY_FILE]

Arguments:
    --log_dir   Name of the Log/ subdirectory to load models from.
                Defaults to the most-recently modified subdirectory of ./Log/.
    --out_dir   Directory to write output files (default: script directory).
    --npy_file  Filename for the saved .npy dictionary (default: eval_ensemble_probs.npy).

Outputs:
    eval_ensemble_probs.npy  — Raw per-model softmax probabilities on Test_60,
                               plus ensembles, labels, protein order, thresholds.
    raw_probabilities.npy    — Pure 2D float32 array (7, N_res) directly loadable
                               with np.load() without allow_pickle=True.
    test_labels.npy          — 1D int32 array (N_res,) of ground truth test labels.
    eval_ensemble_probs.npz  — NPZ archive with individual arrays.
    eval_ensemble_results.txt — Human-readable summary table.

Threshold policy:
    - Fold models (Fold1..5): threshold found by maximising F1 on the model's
      own held-out validation fold (never shown to the model during training).
      Applied unchanged to Test_60. No test labels used.
    - Full_model_40 / Full_model_45: trained on all 334 proteins; no held-out
      fold exists. Mean of the five fold validation thresholds is used.

Fold split reproducibility:
    Importing data_generator seeds numpy with SEED=2024. The script then calls
    KFold(n_splits=5, shuffle=True).split() — matching train.py exactly —
    reproducing the same 5 splits bit-for-bit.
"""

import os
os.environ.setdefault("DGLBACKEND", "pytorch")
import sys
import argparse
import pickle
import warnings
import numpy as np
import pandas as pd
from datetime import datetime

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from sklearn import metrics
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")

# ── Path setup ────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

# Importing data_generator seeds numpy (SEED=2024), torch, and cuda.
from data_generator import (
    SEED, BATCH_SIZE, INPUT_DIM, HIDDEN_DIM, FLITER_DIM,
    OUTPUT_SIZE, DROPOUT, LAYER, ProDataset, graph_collate,
)
from EGNN_model import *
from final_model import FinalModel
from GraphTransformer_Block import *


# ── CLI ───────────────────────────────────────────────────────────────────────
def _latest_log_dir():
    """Return the most recently modified subdirectory of ./Log/."""
    log_root = os.path.join(SCRIPT_DIR, "Log")
    if not os.path.isdir(log_root):
        raise FileNotFoundError(f"Log directory not found: {log_root}")
    subdirs = [
        d for d in os.listdir(log_root)
        if os.path.isdir(os.path.join(log_root, d))
    ]
    if not subdirs:
        raise FileNotFoundError("No subdirectories found in Log/")
    subdirs.sort(key=lambda d: os.path.getmtime(os.path.join(log_root, d)), reverse=True)
    return subdirs[0]


def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate GTE-PPIS ensemble on Test_60 with validation-locked thresholds."
    )
    p.add_argument(
        "--log_dir", type=str, default=None,
        help="Subdirectory name inside ./Log/ (e.g. baseline_run_2). "
             "Defaults to the most-recently modified run.",
    )
    p.add_argument(
        "--out_dir", type=str, default=SCRIPT_DIR,
        help="Directory to write output files (default: script directory).",
    )
    p.add_argument(
        "--npy_file", type=str, default="eval_ensemble_probs.npy",
        help="Filename for the saved .npy probabilities dictionary inside out_dir (default: eval_ensemble_probs.npy).",
    )
    return p.parse_args()


# ── Data helpers ──────────────────────────────────────────────────────────────
DATASET_PATH  = os.path.join(SCRIPT_DIR, "Dataset")
FEATURE_PATH  = "/home/pranav/PRANAV/GTE-PPIS/Feature/"
TEST60_PSEPOS = os.path.join(FEATURE_PATH, "psepos", "Test60_psepos_SC.pkl")
TRAIN_PSEPOS  = os.path.join(FEATURE_PATH, "psepos", "Train335_psepos_SC.pkl")


def load_pkl(name):
    with open(os.path.join(DATASET_PATH, name), "rb") as f:
        return pickle.load(f)


def dict_to_df(dataset_dict, exclude=None):
    exclude = set(exclude or [])
    ids, seqs, labs = [], [], []
    for pid, (seq, lbl) in dataset_dict.items():
        if pid in exclude:
            continue
        ids.append(pid)
        seqs.append(seq)
        labs.append(lbl)
    return pd.DataFrame({"ID": ids, "sequence": seqs, "label": labs})


def make_loader(df, psepos_path, shuffle=False):
    return DataLoader(
        ProDataset(dataframe=df, psepos_path=psepos_path),
        batch_size=BATCH_SIZE, shuffle=shuffle,
        num_workers=2, collate_fn=graph_collate,
    )


# ── Inference ─────────────────────────────────────────────────────────────────
def run_inference(model, loader):
    """
    Returns (y_true, y_pred, protein_order, label_lengths) — all flat per-residue.
    """
    model.eval()
    y_true_all, y_pred_all = [], []
    protein_order, label_lengths = [], []

    with torch.no_grad():
        for batch in loader:
            (seq_names, _, labels, node_features,
             G_batch, adj_matrix, xyz_feats,
             edges, edge_att, edge_feat) = batch

            if torch.cuda.is_available():
                node_features    = Variable(node_features.cuda().float())
                adj_matrix       = Variable(adj_matrix.cuda())
                G_batch.edata["ex"] = Variable(G_batch.edata["ex"].float())
                G_batch          = G_batch.to(torch.device("cuda:0"))
                xyz_feats        = Variable(xyz_feats.cuda().float())
                edges            = Variable(edges.cuda())
                edge_att         = Variable(edge_att.cuda().float())
                edge_feat        = Variable(edge_feat.cuda().float())
                y_true           = Variable(labels.cuda())
            else:
                node_features    = Variable(node_features.float())
                adj_matrix       = Variable(adj_matrix)
                G_batch.edata["ex"] = Variable(G_batch.edata["ex"].float())
                xyz_feats        = Variable(xyz_feats.float())
                edges            = Variable(edges)
                edge_att         = Variable(edge_att.float())
                edge_feat        = Variable(edge_feat.float())
                y_true           = Variable(labels)

            adj_matrix = torch.squeeze(adj_matrix)
            y_true     = torch.squeeze(y_true).long()

            y_pred = model(node_features, xyz_feats, edges, edge_att, edge_feat, adj_matrix)
            prob   = torch.nn.Softmax(dim=1)(y_pred)[:, 1].cpu().detach().numpy()
            truth  = y_true.cpu().detach().numpy()

            y_pred_all.extend(prob.tolist())
            y_true_all.extend(truth.tolist())
            protein_order.append(seq_names[0])
            label_lengths.append(int(truth.shape[0]))

    return y_true_all, y_pred_all, protein_order, label_lengths


# ── Metrics ───────────────────────────────────────────────────────────────────
def auc_metrics(y_true, y_pred):
    auroc = metrics.roc_auc_score(y_true, y_pred)
    prec, rec, _ = metrics.precision_recall_curve(y_true, y_pred)
    auprc = metrics.auc(rec, prec)
    return auroc, auprc


def best_f1_threshold(y_true, y_pred):
    """Sweep 0.00–0.99; return threshold maximising F1."""
    best_f1, best_thr = 0.0, 0.0
    for t in range(0, 100):
        thr = t / 100
        bp = [1 if p >= thr else 0 for p in y_pred]
        f1 = metrics.f1_score(y_true, bp, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, thr
    return best_thr


def threshold_metrics(y_true, y_pred, thr):
    bp = [1 if p >= thr else 0 for p in y_pred]
    return {
        "F1":     metrics.f1_score(y_true, bp, zero_division=0),
        "MCC":    metrics.matthews_corrcoef(y_true, bp),
        "Prec":   metrics.precision_score(y_true, bp, zero_division=0),
        "Recall": metrics.recall_score(y_true, bp, zero_division=0),
        "Acc":    metrics.accuracy_score(y_true, bp),
    }


# ── Model loading ─────────────────────────────────────────────────────────────
def load_model(ckpt_path):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = FinalModel(INPUT_DIM, HIDDEN_DIM, FLITER_DIM, OUTPUT_SIZE, DROPOUT, LAYER)
    if torch.cuda.is_available():
        model.cuda()
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    return model


# ── Formatting ────────────────────────────────────────────────────────────────
def hr(char="─", width=80):
    return char * width


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    # ── Resolve model directory ───────────────────────────────────────────────
    if args.log_dir is None:
        args.log_dir = _latest_log_dir()
    model_dir = os.path.join(SCRIPT_DIR, "Log", args.log_dir, "model")
    if not os.path.isdir(model_dir):
        sys.exit(f"ERROR: Model directory not found: {model_dir}")

    all_files  = sorted(os.listdir(model_dir))
    fold_files = sorted(
        [f for f in all_files if f.startswith("Fold")],
        key=lambda x: int("".join(filter(str.isdigit, x)) or 0)
    )
    full_files = sorted(
        [f for f in all_files if f.startswith("Full")],
        key=lambda x: int("".join(filter(str.isdigit, x)) or 0)
    )

    print(hr("═"))
    print(f"  GTE-PPIS-baseline  |  eval_ensemble.py  |  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(hr("═"))
    print(f"  Log dir   : {args.log_dir}")
    print(f"  Model dir : {model_dir}")
    print(f"  Fold models ({len(fold_files)}): {fold_files}")
    print(f"  Full models ({len(full_files)}): {full_files}")
    dev = ("cuda:" + str(torch.cuda.current_device())) if torch.cuda.is_available() else "cpu"
    print(f"  Device    : {dev}")

    # ── Load Train_334 and reconstruct folds ──────────────────────────────────
    print(f"\n  Loading Train_335 → removing 2j3rA → Train_334...")
    Train_335 = load_pkl("Train_335.pkl")
    if "2j3rA" in Train_335:
        del Train_335["2j3rA"]
    train_df = dict_to_df(Train_335)
    print(f"  Train_334: {len(train_df)} proteins")

    # Reproduces train.py's KFold call exactly.
    # numpy is already seeded at module import (data_generator.py, SEED=2024).
    print(f"\n  Reconstructing KFold(n_splits=5, shuffle=True) with SEED={SEED}...")
    kfold = KFold(n_splits=5, shuffle=True)
    fold_splits = list(kfold.split(train_df["ID"].values, train_df["label"].values))
    for i, (_, val_idx) in enumerate(fold_splits):
        sample = list(train_df["ID"].values[val_idx[:3]])
        print(f"    Fold {i+1}: {len(val_idx)} val proteins  | first 3: {sample}")

    # ── Load Test_60 ──────────────────────────────────────────────────────────
    print(f"\n  Loading Test_60...")
    Test_60 = load_pkl("Test_60.pkl")
    test_df = dict_to_df(Test_60)
    print(f"  Test_60: {len(test_df)} proteins")

    test_loader = make_loader(test_df, TEST60_PSEPOS, shuffle=False)

    # ── STEP 1: Fold models (validation-locked threshold) ────────────────────
    print(f"\n{hr()}")
    print("  STEP 1 — Fold models: threshold locked from held-out validation split")
    print(hr())

    fold_val_thresholds = []
    per_model_results   = {}
    per_model_preds     = {}
    test_true           = None
    test_protein_order  = None
    test_label_lengths  = None

    for fold_i, fname in enumerate(fold_files):
        ckpt = os.path.join(model_dir, fname)
        print(f"\n  [{fname}]")
        model = load_model(ckpt)

        # --- Validation inference (held-out fold) ---
        _, val_idx = fold_splits[fold_i]
        val_df = train_df.iloc[val_idx, :]
        val_loader = make_loader(val_df, TRAIN_PSEPOS, shuffle=False)
        vt, vp, _, _ = run_inference(model, val_loader)
        val_auroc, val_auprc = auc_metrics(vt, vp)
        val_thr = best_f1_threshold(vt, vp)
        val_td  = threshold_metrics(vt, vp, val_thr)
        fold_val_thresholds.append(val_thr)
        print(f"    Val  — AUROC={val_auroc:.4f}  AUPRC={val_auprc:.4f}  "
              f"F1={val_td['F1']:.4f}  MCC={val_td['MCC']:.4f}  "
              f"Thr={val_thr:.2f}")

        # --- Test inference ---
        tt, tp, prot_order, lbl_len = run_inference(model, test_loader)
        if test_true is None:
            test_true          = tt
            test_protein_order = prot_order
            test_label_lengths = lbl_len

        test_auroc, test_auprc = auc_metrics(tt, tp)
        test_td  = threshold_metrics(tt, tp, val_thr)
        oracle_thr = best_f1_threshold(tt, tp)
        oracle_td  = threshold_metrics(tt, tp, oracle_thr)
        print(f"    Test — AUROC={test_auroc:.4f}  AUPRC={test_auprc:.4f}  "
              f"F1={test_td['F1']:.4f}  MCC={test_td['MCC']:.4f}  "
              f"Thr={val_thr:.2f} [val-locked]  "
              f"(oracle F1={oracle_td['F1']:.4f} @{oracle_thr:.2f})")

        per_model_results[fname] = {
            "val_AUROC": val_auroc, "val_AUPRC": val_auprc,
            "val_thr": val_thr,
            "test_AUROC": test_auroc, "test_AUPRC": test_auprc,
            "test_thr_src": "val-locked",
            **{f"test_{k}": v for k, v in test_td.items()},
        }
        per_model_preds[fname] = np.array(tp, dtype=np.float32)

    mean_val_thr = float(np.mean(fold_val_thresholds))
    print(f"\n  Mean fold val threshold: {mean_val_thr:.4f}  "
          f"| per-fold: {[f'{t:.2f}' for t in fold_val_thresholds]}")

    # ── STEP 2: Full models (mean val threshold) ──────────────────────────────
    print(f"\n{hr()}")
    print(f"  STEP 2 — Full models: threshold = mean fold val thr ({mean_val_thr:.2f})")
    print(hr())

    for fname in full_files:
        ckpt = os.path.join(model_dir, fname)
        print(f"\n  [{fname}]")
        model = load_model(ckpt)
        tt, tp, _, _ = run_inference(model, test_loader)
        test_auroc, test_auprc = auc_metrics(tt, tp)
        test_td    = threshold_metrics(tt, tp, mean_val_thr)
        oracle_thr = best_f1_threshold(tt, tp)
        oracle_td  = threshold_metrics(tt, tp, oracle_thr)
        print(f"    Test — AUROC={test_auroc:.4f}  AUPRC={test_auprc:.4f}  "
              f"F1={test_td['F1']:.4f}  MCC={test_td['MCC']:.4f}  "
              f"Thr={mean_val_thr:.2f} [mean-val]  "
              f"(oracle F1={oracle_td['F1']:.4f} @{oracle_thr:.2f})")

        per_model_results[fname] = {
            "val_AUROC": float("nan"), "val_AUPRC": float("nan"),
            "val_thr": mean_val_thr,
            "test_AUROC": test_auroc, "test_AUPRC": test_auprc,
            "test_thr_src": "mean-val",
            **{f"test_{k}": v for k, v in test_td.items()},
        }
        per_model_preds[fname] = np.array(tp, dtype=np.float32)

    # ── STEP 3: Ensembles ─────────────────────────────────────────────────────
    print(f"\n{hr()}")
    print("  STEP 3 — Ensembles (averaged softmax probabilities)")
    print(hr())

    # 5-fold ensemble
    fold_stack = np.stack([per_model_preds[f] for f in fold_files], axis=0)
    ens5 = fold_stack.mean(axis=0)
    ens5_auroc, ens5_auprc = auc_metrics(test_true, ens5.tolist())
    ens5_td_val    = threshold_metrics(test_true, ens5.tolist(), mean_val_thr)
    ens5_oracle_thr = best_f1_threshold(test_true, ens5.tolist())
    ens5_td_oracle = threshold_metrics(test_true, ens5.tolist(), ens5_oracle_thr)

    print(f"\n  Ensemble-5fold:")
    print(f"    AUROC={ens5_auroc:.4f}  AUPRC={ens5_auprc:.4f}")
    print(f"    [val thr={mean_val_thr:.2f}]  "
          f"F1={ens5_td_val['F1']:.4f}  MCC={ens5_td_val['MCC']:.4f}  "
          f"Prec={ens5_td_val['Prec']:.4f}  Rec={ens5_td_val['Recall']:.4f}")
    print(f"    [oracle thr={ens5_oracle_thr:.2f}]  "
          f"F1={ens5_td_oracle['F1']:.4f}  MCC={ens5_td_oracle['MCC']:.4f}  "
          f"Prec={ens5_td_oracle['Prec']:.4f}  Rec={ens5_td_oracle['Recall']:.4f}")

    # All-7 ensemble
    all_ordered = fold_files + full_files
    all_stack = np.stack([per_model_preds[f] for f in all_ordered], axis=0)
    ens7 = all_stack.mean(axis=0)
    ens7_auroc, ens7_auprc = auc_metrics(test_true, ens7.tolist())
    ens7_td_val    = threshold_metrics(test_true, ens7.tolist(), mean_val_thr)
    ens7_oracle_thr = best_f1_threshold(test_true, ens7.tolist())
    ens7_td_oracle = threshold_metrics(test_true, ens7.tolist(), ens7_oracle_thr)

    print(f"\n  Ensemble-7all:")
    print(f"    AUROC={ens7_auroc:.4f}  AUPRC={ens7_auprc:.4f}")
    print(f"    [val thr={mean_val_thr:.2f}]  "
          f"F1={ens7_td_val['F1']:.4f}  MCC={ens7_td_val['MCC']:.4f}  "
          f"Prec={ens7_td_val['Prec']:.4f}  Rec={ens7_td_val['Recall']:.4f}")
    print(f"    [oracle thr={ens7_oracle_thr:.2f}]  "
          f"F1={ens7_td_oracle['F1']:.4f}  MCC={ens7_td_oracle['MCC']:.4f}  "
          f"Prec={ens7_td_oracle['Prec']:.4f}  Rec={ens7_td_oracle['Recall']:.4f}")

    # ── Summary table ─────────────────────────────────────────────────────────
    W = 102
    print(f"\n{hr('═', W)}")
    print(f"  SUMMARY — Test_60  |  log_dir={args.log_dir}")
    print(hr("═", W))
    hdr = f"  {'Model':<30}  {'AUROC':>7}  {'AUPRC':>7}  {'MCC':>7}  {'F1':>7}  {'Prec':>7}  {'Rec':>7}  {'Thr':>5}  Source"
    print(hdr)
    print(f"  {hr('-', W-2)}")
    for fname in all_ordered:
        r = per_model_results[fname]
        print(f"  {fname:<30}  "
              f"{r['test_AUROC']:>7.4f}  {r['test_AUPRC']:>7.4f}  "
              f"{r['test_MCC']:>7.4f}  {r['test_F1']:>7.4f}  "
              f"{r['test_Prec']:>7.4f}  {r['test_Recall']:>7.4f}  "
              f"{r['val_thr']:>5.2f}  {r['test_thr_src']}")
    print(f"  {hr('-', W-2)}")
    print(f"  {'Ensemble-5fold [val-locked]':<30}  "
          f"{ens5_auroc:>7.4f}  {ens5_auprc:>7.4f}  "
          f"{ens5_td_val['MCC']:>7.4f}  {ens5_td_val['F1']:>7.4f}  "
          f"{ens5_td_val['Prec']:>7.4f}  {ens5_td_val['Recall']:>7.4f}  "
          f"{mean_val_thr:>5.2f}  mean-val")
    print(f"  {'Ensemble-5fold [oracle]':<30}  "
          f"{ens5_auroc:>7.4f}  {ens5_auprc:>7.4f}  "
          f"{ens5_td_oracle['MCC']:>7.4f}  {ens5_td_oracle['F1']:>7.4f}  "
          f"{ens5_td_oracle['Prec']:>7.4f}  {ens5_td_oracle['Recall']:>7.4f}  "
          f"{ens5_oracle_thr:>5.2f}  oracle (test-set — informational only)")
    print(f"  {'Ensemble-7all  [val-locked]':<30}  "
          f"{ens7_auroc:>7.4f}  {ens7_auprc:>7.4f}  "
          f"{ens7_td_val['MCC']:>7.4f}  {ens7_td_val['F1']:>7.4f}  "
          f"{ens7_td_val['Prec']:>7.4f}  {ens7_td_val['Recall']:>7.4f}  "
          f"{mean_val_thr:>5.2f}  mean-val")
    print(f"  {'Ensemble-7all  [oracle]':<30}  "
          f"{ens7_auroc:>7.4f}  {ens7_auprc:>7.4f}  "
          f"{ens7_td_oracle['MCC']:>7.4f}  {ens7_td_oracle['F1']:>7.4f}  "
          f"{ens7_td_oracle['Prec']:>7.4f}  {ens7_td_oracle['Recall']:>7.4f}  "
          f"{ens7_oracle_thr:>5.2f}  oracle (test-set — informational only)")
    print(hr("═", W))

    print("""
  THRESHOLD NOTE
  ─────────────
  [val-locked]  threshold selected by maximising F1 on the model's own held-out
                validation split (data the model never saw during training). Use
                these numbers when comparing to other methods. AUC/AUPRC are the
                same regardless of threshold choice.

  [oracle]      threshold swept over the test set itself — inflates F1/MCC.
                Shown for reference; matches what the original test.py reports.

  [mean-val]    mean of the 5 fold thresholds, applied to Full models and ensembles.
""")

    # ── Save .npy and .npz outputs ────────────────────────────────────────────
    os.makedirs(args.out_dir, exist_ok=True)

    save_dict = {
        "models":              np.array(all_ordered, dtype=str),
        "raw_probs":           all_stack.astype(np.float32),
        "ens5_pred":           ens5.astype(np.float32),
        "ens7_pred":           ens7.astype(np.float32),
        "labels":              np.array(test_true, dtype=np.int32),
        "protein_order":       np.array(test_protein_order, dtype=str),
        "label_lengths":       np.array(test_label_lengths, dtype=np.int32),
        "mean_val_thr":        np.float32(mean_val_thr),
        "fold_val_thresholds": np.array(fold_val_thresholds, dtype=np.float32),
    }
    for fname, arr in per_model_preds.items():
        save_dict[fname.replace(".pkl", "")] = arr.astype(np.float32)

    # 1. Primary .npy file: dict containing per-model probs, ensembles, labels & metadata
    npy_path = os.path.join(args.out_dir, args.npy_file)
    np.save(npy_path, save_dict, allow_pickle=True)
    print(f"  Raw probabilities (.npy dict) saved → {npy_path}")

    # 2. Pure 2D array .npy file: (7, 13144) float32 ndarray (loadable with np.load without pickle)
    raw_array_path = os.path.join(args.out_dir, "raw_probabilities.npy")
    np.save(raw_array_path, all_stack.astype(np.float32))
    print(f"  Raw probabilities (7x{len(test_true)} ndarray) saved → {raw_array_path}")

    # 3. Test ground truth labels .npy file: (13144,) int32
    labels_path = os.path.join(args.out_dir, "test_labels.npy")
    np.save(labels_path, np.array(test_true, dtype=np.int32))
    print(f"  Test labels ({len(test_true)} int32 ndarray) saved → {labels_path}")

    # 4. NPZ archive
    npz_path = os.path.join(args.out_dir, "eval_ensemble_probs.npz")
    np.savez(npz_path, **save_dict)
    print(f"  Probabilities (.npz archive) saved → {npz_path}")
    print(f"  NPZ/dict keys: {list(save_dict.keys())}")
    print(f"  Labels: {len(test_true)} residues across {len(test_protein_order)} proteins")

    # ── Save text report ──────────────────────────────────────────────────────
    txt_path = os.path.join(args.out_dir, "eval_ensemble_results.txt")
    with open(txt_path, "w") as fh:
        fh.write(f"GTE-PPIS eval_ensemble.py | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        fh.write(f"log_dir: {args.log_dir}\n")
        fh.write(f"mean_val_thr: {mean_val_thr:.4f}\n")
        fh.write(f"fold_val_thrs: {[round(t,2) for t in fold_val_thresholds]}\n\n")
        fh.write(f"{'Model':<30}  AUROC   AUPRC   MCC     F1      Prec    Rec     Thr   Source\n")
        fh.write("-" * 100 + "\n")
        for fname in all_ordered:
            r = per_model_results[fname]
            fh.write(f"{fname:<30}  {r['test_AUROC']:.4f}  {r['test_AUPRC']:.4f}  "
                     f"{r['test_MCC']:.4f}  {r['test_F1']:.4f}  "
                     f"{r['test_Prec']:.4f}  {r['test_Recall']:.4f}  "
                     f"{r['val_thr']:.2f}  {r['test_thr_src']}\n")
        fh.write("-" * 100 + "\n")
        for label, auroc, auprc, td, thr, src in [
            ("Ensemble-5fold [val-locked]",
             ens5_auroc, ens5_auprc, ens5_td_val, mean_val_thr, "mean-val"),
            ("Ensemble-5fold [oracle]",
             ens5_auroc, ens5_auprc, ens5_td_oracle, ens5_oracle_thr, "oracle"),
            ("Ensemble-7all  [val-locked]",
             ens7_auroc, ens7_auprc, ens7_td_val, mean_val_thr, "mean-val"),
            ("Ensemble-7all  [oracle]",
             ens7_auroc, ens7_auprc, ens7_td_oracle, ens7_oracle_thr, "oracle"),
        ]:
            fh.write(f"{label:<30}  {auroc:.4f}  {auprc:.4f}  "
                     f"{td['MCC']:.4f}  {td['F1']:.4f}  "
                     f"{td['Prec']:.4f}  {td['Recall']:.4f}  "
                     f"{thr:.2f}  {src}\n")
    print(f"  Text report saved  → {txt_path}\n")


if __name__ == "__main__":
    main()
