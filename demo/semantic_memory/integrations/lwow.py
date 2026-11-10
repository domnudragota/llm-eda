from __future__ import annotations

import importlib.util
from functools import lru_cache
from pathlib import Path
import sys

import pandas as pd

from ..config import DEFAULT_REPO


@lru_cache(maxsize=4)
def load_original_functions(repo: Path):
    file = Path(repo).resolve() / 'reproducibility' / 'FA_Functions.py'
    if not file.is_file():
        raise FileNotFoundError(
            f'Original researcher source not present: {file}\n'
            'Run scripts/setup_lwow.ps1 first.'
        )

    module_name = 'authors_original_FA_Functions'
    spec = importlib.util.spec_from_file_location(module_name, str(file))
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Cannot import source file {file}')

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    return module


def official_graph_path(repo: Path, model: str = 'Haiku') -> Path:
    if model == 'Humans':
        return Path(repo) / 'reproducibility' / 'data' / 'graphs' / 'edge_lists' / 'FA_Humans_edgelist.csv'

    if model not in {'Haiku', 'Mistral', 'Llama3'}:
        raise ValueError('Unknown LWOW/SWOW association dataset')

    return Path(repo) / 'LWOW_datasets' / 'graphs' / 'edge_lists' / f'FA_{model}_edgelist.csv'


def official_gender_matrix_path(repo: Path, model: str = 'Haiku') -> Path:
    if model not in {'Haiku', 'Mistral', 'Llama3', 'Humans'}:
        raise ValueError('Unknown original LWOW dataset')

    return Path(repo) / 'reproducibility' / 'data' / 'LDT_analyses' / 'FA_matrices' / f'{model}_gender.csv'


def load_normalized_activation(repo: Path, model: str) -> pd.DataFrame:
    raw = pd.read_csv(official_gender_matrix_path(repo, model))

    return load_original_functions(repo).normalizeDF(raw, normalize_rows=True)
