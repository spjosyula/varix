"""varix.analysis — localizer and classifier registry.

Depends only on `varix.core` types and the outputs of the execution layer.
Does not import adapters or any I/O layer.
"""

from varix.analysis.localizer import Localizer
from varix.analysis.orchestration import AnalysisResult, analyze
from varix.analysis.registry import Classifier, ClassifierRegistry

__all__ = [
    "AnalysisResult",
    "Classifier",
    "ClassifierRegistry",
    "Localizer",
    "analyze",
]
