from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class DataIngestionArtifact:
    data_zip_file_path:str
    feature_store_path:str


@dataclass
class DataValidationArtifact:
    validation_status: bool


@dataclass
class ModelTrainerArtifact:
    trained_model_file_path: str


@dataclass
class ModelEvaluationArtifact:
    is_model_accepted: bool
    evaluated_model_path: str
    metrics_file_path: str
    model_metrics: Dict[str, Any] = None
    trained_model_path: str = None