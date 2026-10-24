"""
src/data/__init__.py
Módulos de ingeniería y preparación de datos para Trabajo 2 (¿me prestas?).
Curso: Redes Neuronales y Algoritmos Bioinspirados - Universidad Nacional de Colombia.
"""

from .clean import clean_loans, get_pd_target_mapping
from .leakage import remove_leakage, get_leakage_columns, generate_leakage_report
from .lgd import compute_lgd
from .split import temporal_split, random_split
from .pipeline import build_preprocessor, transform_split
from .loader import load_splits

__all__ = [
    "clean_loans",
    "get_pd_target_mapping",
    "remove_leakage",
    "get_leakage_columns",
    "generate_leakage_report",
    "compute_lgd",
    "temporal_split",
    "random_split",
    "build_preprocessor",
    "transform_split",
    "load_splits",
]
