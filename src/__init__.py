"""SciCite Citation Intent Classification package."""

from src.data import (
    CLASS_NAMES,
    LABEL2ID,
    ID2LABEL,
    CANONICAL_SECTIONS,
    normalize_section_name,
    build_representations,
    get_data_splits,
)
from src.utils import set_seed, compute_classification_metrics, plot_confusion_matrices
from src.models import train_baseline, train_transformer
