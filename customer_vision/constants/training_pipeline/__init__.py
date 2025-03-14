ARTIFACTS_DIR: str = "artifacts"

"""
Data Ingestion related constants start with DATA_INGESTION VAR NAME
"""
DATA_INGESTION_DIR_NAME: str = "data_ingestion"

DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"

DATA_DOWNLOAD_URL: str = "https://universe.roboflow.com/ds/xotN3EeKkU?key=ct69OU65AU"



"""
Data Validation related constants start with DATA_VALIDATION VAR NAME
"""

DATA_VALIDATION_DIR_NAME: str = "data_validation"

DATA_VALIDATION_STATUS_FILE = 'status.txt'

DATA_VALIDATION_ALL_REQUIRED_FILES = ["train", "valid", "data.yaml"]


"""
MODEL TRAINER related constants start with MODEL_TRAINER VAR NAME
"""
MODEL_TRAINER_DIR_NAME: str = "model_trainer"

MODEL_TRAINER_PRETRAINED_WEIGHT_NAME: str = "yolov8s.pt"

MODEL_TRAINER_NO_EPOCHS: int = 1

MODEL_TRAINER_BATCH_SIZE: int = 8

MODEL_TRAINER_IMAGE_SIZE: int = 416

DETECTION_CLASSES = ["person"]

MODEL_EVALUATION_THRESHOLD: float = 0.25
MODEL_EVALUATION_IOU_THRESHOLD: float = 0.5
MODEL_EVALUATION_MIN_MAP_THRESHOLD: float = 0.5

