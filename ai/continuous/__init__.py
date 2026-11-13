"""
Signova AI - Continuous ISL Sentence Translation Package.
Wraps and re-exports existing iSign T5 continuous translation modules.
Preserves existing T5 implementation without modification.
"""

"""
Signova AI - Continuous ISL Sentence Translation Package.
Wraps and re-exports continuous translation modules and Bi-GRU + CTC baseline.
"""

from ai.continuous.bigru_ctc import (
    BiGRUCTCModel,
    ContinuousISLDataset,
    collate_continuous_fn,
    compute_subsampled_lengths,
    decode_greedy,
    compute_levenshtein,
    GLOSS_TO_ID,
    ID_TO_GLOSS,
    NUM_CLASSES,
    BLANK_INDEX,
)
from ai.continuous.temporal_features import extract_temporal_features
from ai.continuous.sentence_dataset import (
    ISLCSLTRSentenceDataset,
    collate_sentence_batch,
    compute_smoothed_class_weights,
)
from ai.continuous.sentence_classifier import (
    TemporalSentenceClassifier,
    TemporalAttention,
    TemporalConv1DBlock,
)

# Optional legacy T5 wrappers
try:
    from ai.inference.model_loader import get_model as get_loaded_model, get_tokenizer as get_loaded_tokenizer, load_model, unload_model
except ImportError:
    pass

try:
    from ai.inference.pose_translator import PoseTranslator, InferenceResult as TranslationOutput
except ImportError:
    pass

try:
    from ai.pose_extraction.pose_generator import video_to_pose_file
except (ImportError, ModuleNotFoundError):
    pass

try:
    from ai.preprocessing.pose_preprocessor import (
        add_motion_features,
        normalize_pose,
        pad_or_truncate_pose,
        preprocess_pose,
    )
except ImportError:
    pass

__all__ = [
    "BiGRUCTCModel",
    "ContinuousISLDataset",
    "collate_continuous_fn",
    "compute_subsampled_lengths",
    "decode_greedy",
    "compute_levenshtein",
    "GLOSS_TO_ID",
    "ID_TO_GLOSS",
    "NUM_CLASSES",
    "BLANK_INDEX",
    "extract_temporal_features",
    "ISLCSLTRSentenceDataset",
    "collate_sentence_batch",
    "compute_smoothed_class_weights",
    "TemporalSentenceClassifier",
    "TemporalAttention",
    "TemporalConv1DBlock",
]
