#!/usr/bin/env python3
"""
Standard CEBRA-Time pipeline: load CSV as multi-session segments, train, visualize.
Input CSV: embryo_id, time_step, [phase], and remaining columns as latent features (e.g. z0, z1, ...).
"""
import os
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# PyTorch 2.6+ weights_only; CEBRA checkpoints need weights_only=False
import torch
_orig_load = torch.load
torch.load = lambda *a, **k: _orig_load(*a, **{**k, "weights_only": False})

import cebra
from cebra import CEBRA

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_ivf_csv_as_sessions(csv_path, min_length=10, meta_cols=None):
    df = pd.read_csv(csv_path)
    df = df.sort_values(["embryo_id", "time_step"]).reset_index(drop=True)

    if meta_cols is None:
        meta_cols = {"index", "embryo_id", "time_step", "TE", "ICM", "phase"}
    meta_cols = {c for c in meta_cols if c in df.columns}
    feature_cols = [c for c in df.columns if c not in meta_cols]
    if not feature_cols:
        raise ValueError("No feature columns found. Ensure CSV has latent columns (e.g. z0, z1, ...).")

    X = df[feature_cols].astype(np.float32).values
    embryo_id = df["embryo_id"].values
    time_step = df["time_step"].astype(np.float32).values
    phase = df["phase"].astype(str).values if "phase" in df.columns else np.full(len(df), "", dtype=object)

    sessions = []
    time_labels = []
    session_meta = []

    for eid in pd.unique(embryo_id):
        idx = np.where(embryo_id == eid)[0]
        if len(idx) < min_length:
            continue
        sessions.append(X[idx])
        time_labels.append(time_step[idx].reshape(-1, 1))
        session_meta.append((eid, idx))

    return sessions, time_labels, session_meta, phase


def train_cebra_time(
    sessions,
    time_labels,
    out_path="cebra_time_model.pt",
    max_iterations=20000,
    time_offsets=10,
    batch_size=32,
    lr=1e-3,
    out_dim=16,
    hidden=128,
):
    model = CEBRA(
        model_architecture="offset10-model",
        batch_size=batch_size,
        learning_rate=lr,
        temperature=1,
        output_dimension=out_dim,
        num_hidden_units=hidden,
        max_iterations=max_iterations,
        distance="cosine",
        conditional="time",
        device="cuda_if_available",
        verbose=True,
        time_offsets=time_offsets,
    )
    model.fit(sessions, time_labels)
    model.save(out_path)
    return model


def visualize_by_embryo(model, sessions, session_meta, out_dir="embryo_embeddings", max_plots=50):
    os.makedirs(out_dir, exist_ok=True)
    n = min(len(sessions), max_plots)

    for k in range(n):
        eid, idx = session_meta[k]
        X = sessions[k]
        emb = model.transform(X)
        t = np.linspace(0, 1, len(X), dtype=np.float32)

        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111, projection="3d")
        cebra.plot_embedding(
            ax=ax,
            embedding=emb,
            embedding_labels=t,
            markersize=1,
            title=f"Embryo {eid} (n={len(X)})",
        )
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"embryo_{eid}.png"), dpi=150)
        plt.close()

    print(f"Saved {n} embryo plots to: {out_dir}/")


def visualize_all_points(model, sessions, out_path="embedding_all.png", max_points=200000):
    X = np.concatenate(sessions, axis=0)
    if len(X) > max_points:
        rng = np.random.default_rng(0)
        X = X[rng.choice(len(X), size=max_points, replace=False)]

    emb = model.transform(X)
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    cebra.plot_embedding(
        ax=ax,
        embedding=emb,
        embedding_labels=np.zeros(len(emb)),
        markersize=1,
        title="All points",
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


def run(
    csv_path,
    out_path="cebra_time_model.pt",
    max_iterations=20000,
    time_offsets=10,
    batch_size=32,
    lr=1e-3,
    out_dim=16,
    hidden=128,
    min_session_length=10,
    embryo_embeddings_dir="embryo_embeddings",
    embedding_all_path="embedding_all.png",
    max_plots=50,
):
    sessions, time_labels, session_meta, phase = load_ivf_csv_as_sessions(
        csv_path, min_length=min_session_length
    )
    if not sessions:
        raise ValueError("No sessions with enough frames. Check min_session_length and CSV.")

    model = train_cebra_time(
        sessions,
        time_labels,
        out_path=out_path,
        max_iterations=max_iterations,
        time_offsets=time_offsets,
        batch_size=batch_size,
        lr=lr,
        out_dim=out_dim,
        hidden=hidden,
    )

    visualize_all_points(model, sessions, out_path=embedding_all_path)
    visualize_by_embryo(
        model, sessions, session_meta, out_dir=embryo_embeddings_dir, max_plots=max_plots
    )
    return model


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="CEBRA-Time: load CSV as sessions, train, visualize.")
    ap.add_argument("csv_path", help="CSV with embryo_id, time_step, [phase], and feature columns (e.g. z0, z1, ...)")
    ap.add_argument("--out_model", default="cebra_time_model.pt", help="Model checkpoint path")
    ap.add_argument("--max_iterations", type=int, default=20000)
    ap.add_argument("--time_offsets", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=32, help="Smaller (e.g. 32) to avoid OOM with 512-d input")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out_dim", type=int, default=16)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--min_session_length", type=int, default=10)
    ap.add_argument("--embryo_embeddings_dir", default="embryo_embeddings")
    ap.add_argument("--embedding_all", default="embedding_all.png")
    ap.add_argument("--max_plots", type=int, default=50)
    args = ap.parse_args()

    run(
        args.csv_path,
        out_path=args.out_model,
        max_iterations=args.max_iterations,
        time_offsets=args.time_offsets,
        batch_size=args.batch_size,
        lr=args.lr,
        out_dim=args.out_dim,
        hidden=args.hidden,
        min_session_length=args.min_session_length,
        embryo_embeddings_dir=args.embryo_embeddings_dir,
        embedding_all_path=args.embedding_all,
        max_plots=args.max_plots,
    )
