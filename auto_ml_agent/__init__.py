from .core import (
    ExperimentAgent,
    Task,
    TaskType,
    EnvironmentManager,
    TrainingExecutor,
    TrainingMonitor,
    AnomalyType,
    AnomalyRecord,
    ReportGenerator,
    ExperimentResult,
    ExperimentPackager,
    NotificationManager,
    NotificationChannel,
)
from .templates import (
    ExperimentTemplate,
    TemplateRegistry,
    ImageClassificationTemplate,
    TextClassificationTemplate,
    ObjectDetectionTemplate,
)
from .utils import (
    load_config,
    get_config_value,
    get_logger,
)

__version__ = "0.1.0"
__author__ = "AutoML Agent Team"

__all__ = [
    "__version__",
    "__author__",
    "ExperimentAgent",
    "Task",
    "TaskType",
    "EnvironmentManager",
    "TrainingExecutor",
    "TrainingMonitor",
    "AnomalyType",
    "AnomalyRecord",
    "ReportGenerator",
    "ExperimentResult",
    "ExperimentPackager",
    "NotificationManager",
    "NotificationChannel",
    "ExperimentTemplate",
    "TemplateRegistry",
    "ImageClassificationTemplate",
    "TextClassificationTemplate",
    "ObjectDetectionTemplate",
    "load_config",
    "get_config_value",
    "get_logger",
]
