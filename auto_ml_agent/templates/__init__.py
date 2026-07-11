from .base import ExperimentTemplate, TemplateRegistry
from .image_classification import ImageClassificationTemplate
from .text_classification import TextClassificationTemplate
from .object_detection import ObjectDetectionTemplate

__all__ = [
    "ExperimentTemplate",
    "TemplateRegistry",
    "ImageClassificationTemplate",
    "TextClassificationTemplate",
    "ObjectDetectionTemplate",
]
