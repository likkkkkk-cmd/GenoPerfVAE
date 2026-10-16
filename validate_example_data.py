#!/usr/bin/env python3
"""Check two user-provided local Wheat2000 length inputs (no model training).

WARNING: pandas.read_pickle() can execute code embedded in untrusted pickle
files. Only run against the local data lawfully obtained from a trusted original source; data are not bundled.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', type=Path, default=Path('example_data'))
    p.add_argument('--expected-samples', type=int, default=None,
                   help='Optional expected number of rows; default: do not assume')
    args = p.parse_args()
    g = args.data_dir / '2000gene.pkl'
    y = args.data_dir / '2000_3_phe_length.txt'
    for f in (g, y):
        if not f.is_file():
            p.error(f'Missing required locally obtained input: {f}')
    print('CAUTION: Loading a pickle file. Run only on trusted, locally obtained data.')
    obj = pd.read_pickle(g)
    if isinstance(obj, pd.DataFrame):
        matrix = obj.to_numpy()
    elif isinstance(obj, pd.Series):
        arr = obj.to_numpy()
        matrix = np.stack(arr) if (arr.ndim == 1 and len(arr) and np.ndim(arr[0])>0) else arr
    else:
        matrix = np.asarray(obj)
    if matrix.ndim != 2:
        raise ValueError(f'Genotype matrix must be 2-D; found {matrix.shape}')
    try:
        x = matrix.astype(np.float32, copy=False)
    except (ValueError, TypeError) as exc:
        raise ValueError('Genotype matrix must be numeric') from exc
    if np.isinf(x).any():
        raise ValueError('Genotype matrix contains infinite values')
    phenotypes = []
    with y.open('r', encoding='utf-8-sig') as f:
        for line_no, line in enumerate(f, 1):
            stripped = line.strip()
            if not stripped:
                continue
            tokens = stripped.replace(',', ' ').replace(';', ' ').split()
            found = None
            for tok in tokens:
                try:
                    found = float(tok)
                    break
                except ValueError:
                    pass
            if found is None:
                raise ValueError(f'No numeric phenotype value on line {line_no}')
            if not np.isfinite(found):
                raise ValueError(f'Non-finite phenotype value on line {line_no}')
            phenotypes.append(found)
    n = len(phenotypes)
    print(f'Genotype: {g.name} | shape={x.shape} | NaNs={int(np.isnan(x).sum())}')
    print(f'Phenotype: {y.name} | numeric rows={n}')
    if x.shape[0] != n:
        raise ValueError(f'Row-count mismatch: genotype={x.shape[0]}, phenotype={n}')
    if args.expected_samples is not None and x.shape[0] != args.expected_samples:
        raise ValueError(f'Expected {args.expected_samples} samples, found {x.shape[0]}')
    print('PASS: Basic data format and row-count consistency.')
    print('NOTE: Matching row counts do not verify accession identity or order.')


if __name__ == '__main__':
    main()
