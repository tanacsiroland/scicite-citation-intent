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
from src.utils import (
    set_seed,
    compute_classification_metrics,
    plot_confusion_matrices,
    display_saved_figure,
    plot_class_distribution,
    plot_canonical_section_contingency,
    plot_prediction_probabilities,
)
from src.models import (
    train_baseline,
    train_transformer,
    save_transformer_model,
    save_baseline_pipeline,
    push_model_to_hub,
)
