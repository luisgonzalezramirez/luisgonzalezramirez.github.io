#!/usr/bin/env python3
"""Inspect a Lallement/GalaxyMap HDF5 dust cube without loading it fully."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


def describe_attrs(obj, indent: str = "") -> None:
    if not obj.attrs:
        return
    print(f"{indent}attrs:")
    for key, val in obj.attrs.items():
        try:
            if hasattr(val, "shape") and val.size > 12:
                shown = np.asarray(val).ravel()[:12]
                print(f"{indent}  {key}: {shown!r} ... shape={val.shape}")
            else:
                print(f"{indent}  {key}: {val!r}")
        except Exception:
            print(f"{indent}  {key}: {val}")


def walk_h5(path: Path) -> list[tuple[str, tuple[int, ...], str]]:
    datasets: list[tuple[str, tuple[int, ...], str]] = []
    with h5py.File(path, "r") as h5:
        print(f"FILE: {path}")
        describe_attrs(h5)
        print("\nTREE:")

        def visitor(name: str, obj) -> None:
            depth = name.count("/")
            indent = "  " * depth
            if isinstance(obj, h5py.Group):
                print(f"{indent}[G] /{name}")
                describe_attrs(obj, indent + "  ")
            elif isinstance(obj, h5py.Dataset):
                shape = tuple(obj.shape)
                dtype = str(obj.dtype)
                chunks = obj.chunks
                compression = obj.compression
                print(f"{indent}[D] /{name} shape={shape} dtype={dtype} chunks={chunks} compression={compression}")
                describe_attrs(obj, indent + "  ")
                datasets.append((name, shape, dtype))

        h5.visititems(visitor)

    return datasets


def sample_dataset(path: Path, dataset_name: str) -> None:
    with h5py.File(path, "r") as h5:
        ds = h5[dataset_name]
        print(f"\nSAMPLE: /{dataset_name}")
        print(f"shape={ds.shape} dtype={ds.dtype}")
        if ds.ndim == 0:
            arr = np.asarray(ds[()])
        elif ds.ndim == 1:
            n = min(ds.shape[0], 20)
            arr = np.asarray(ds[:n])
        elif ds.ndim == 2:
            arr = np.asarray(ds[:min(ds.shape[0], 8), :min(ds.shape[1], 8)])
        elif ds.ndim == 3:
            i = ds.shape[0] // 2
            arr = np.asarray(ds[i, :min(ds.shape[1], 8), :min(ds.shape[2], 8)])
            print(f"central slice index along axis 0: {i}")
        else:
            sl = tuple(slice(0, min(s, 3)) for s in ds.shape)
            arr = np.asarray(ds[sl])
        print(arr)
        if np.issubdtype(arr.dtype, np.number):
            full_sample = None
            if ds.ndim == 3:
                sl = tuple(slice(0, min(s, 32)) for s in ds.shape)
                full_sample = np.asarray(ds[sl], dtype=float)
            else:
                full_sample = np.asarray(arr, dtype=float)
            finite = full_sample[np.isfinite(full_sample)]
            if finite.size:
                print("sample stats:", {
                    "min": float(np.nanmin(finite)),
                    "p50": float(np.nanpercentile(finite, 50)),
                    "p95": float(np.nanpercentile(finite, 95)),
                    "max": float(np.nanmax(finite)),
                })


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("h5", type=Path, help="Path to map3D_GAIAdr2_feb2019.h5")
    ap.add_argument("--sample", default=None, help="Dataset path to sample, e.g. dust/map")
    args = ap.parse_args()

    datasets = walk_h5(args.h5)
    numeric_3d = [(n, s, d) for n, s, d in datasets if len(s) == 3 and not d.startswith("|")]
    if numeric_3d:
        print("\nNUMERIC 3D DATASET CANDIDATES:")
        for n, s, d in numeric_3d:
            print(f"  /{n} shape={s} dtype={d}")
    if args.sample:
        sample_dataset(args.h5, args.sample.strip("/"))


if __name__ == "__main__":
    main()
