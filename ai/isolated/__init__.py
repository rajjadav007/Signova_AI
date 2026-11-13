"""
Signova AI - Individual ISL Sign Recognition Package.
Independent module decoupled from Continuous Sentence Translation.
"""

from ai.isolated.config.isolated_settings import ISOLATED_CLASSES, NUM_CLASSES
from ai.isolated.inference.classifier import IsolatedSignRecognizer

__all__ = ["ISOLATED_CLASSES", "NUM_CLASSES", "IsolatedSignRecognizer"]
