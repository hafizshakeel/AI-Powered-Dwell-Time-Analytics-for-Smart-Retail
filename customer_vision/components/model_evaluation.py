import os
import sys
import yaml
from ultralytics import YOLO
from customer_vision.logger import logging
from customer_vision.exception import AppException
from customer_vision.entity.config_entity import ModelEvaluationConfig
from customer_vision.entity.artifacts_entity import ModelTrainerArtifact, ModelEvaluationArtifact


class ModelEvaluation:
    def __init__(
        self,
        model_evaluation_config: ModelEvaluationConfig,
        model_trainer_artifact: ModelTrainerArtifact
    ):
        try:
            self.model_evaluation_config = model_evaluation_config
            self.model_trainer_artifact = model_trainer_artifact
        except Exception as e:
            raise AppException(e, sys)
    
    def evaluate_model(self) -> dict:
        """
        Evaluate the trained model on validation data
        """
        try:
            logging.info("Starting model evaluation")
            
            # Load the trained model
            model = YOLO(self.model_trainer_artifact.trained_model_file_path)
            
            # Get the path to data.yaml from model trainer
            data_yaml_path = os.path.join("artifacts", "model_trainer", "data.yaml")
            
            if not os.path.exists(data_yaml_path):
                logging.warning(f"data.yaml not found at {data_yaml_path}, trying to find it in feature store")
                # Try to find it in the feature store
                feature_store_yaml_path = os.path.join("artifacts", "data_ingestion", "feature_store", "data.yaml")
                if os.path.exists(feature_store_yaml_path):
                    logging.info(f"Using data.yaml from feature store: {feature_store_yaml_path}")
                    data_yaml_path = feature_store_yaml_path
                else:
                    raise FileNotFoundError(f"data.yaml not found at {data_yaml_path} or {feature_store_yaml_path}")
            
            logging.info(f"Using data.yaml from: {data_yaml_path}")
            
            # Run validation on the validation dataset
            results = model.val(
                data=data_yaml_path,
                conf=self.model_evaluation_config.threshold,
                iou=self.model_evaluation_config.iou_threshold,
                verbose=True
            )
            
            # Extract metrics using the correct attributes
            metrics_dict = results.results_dict
            
            # Log available keys for debugging
            logging.info(f"Available metrics keys: {list(metrics_dict.keys())}")
            
            # Initialize metrics dictionary with safe defaults
            metrics = {
                "mAP50": results.maps[0] if len(results.maps) > 0 else 0,  # mAP at IoU 0.5
                "fitness": results.fitness  # Overall fitness score
            }
            
            # Safely try to get mAP50-95 if available
            try:
                if len(results.maps) > 1:
                    metrics["mAP50-95"] = results.maps[1]
                else:
                    metrics["mAP50-95"] = results.maps[0]  # For single class, use mAP50
                    logging.info("Using mAP50 as mAP50-95 for single class evaluation")
            except Exception as e:
                logging.warning(f"Could not get mAP50-95: {str(e)}")
                metrics["mAP50-95"] = metrics["mAP50"]
            
            # Add precision and recall metrics
            try:
                box_mean_results = results.box.mean_results()
                if len(box_mean_results) >= 3:
                    metrics["precision"] = float(box_mean_results[0])
                    metrics["recall"] = float(box_mean_results[1])
                else:
                    # Fallback to metrics from results_dict if available
                    metrics["precision"] = float(metrics_dict.get('metrics/precision(B)', 0))
                    metrics["recall"] = float(metrics_dict.get('metrics/recall(B)', 0))
            except Exception as e:
                logging.warning(f"Error extracting precision/recall metrics: {str(e)}")
                metrics["precision"] = 0
                metrics["recall"] = 0
            
            logging.info(f"Model evaluation metrics: {metrics}")
            return metrics
            
        except Exception as e:
            logging.error(f"Error during model evaluation: {str(e)}")
            raise AppException(e, sys)
    
    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        try:
            logging.info("Initiating model evaluation")
            
            # Create evaluation directory
            os.makedirs(self.model_evaluation_config.model_evaluation_dir, exist_ok=True)
            
            # Evaluate model
            metrics = self.evaluate_model()
            
            # Save metrics to file
            metrics_file_path = os.path.join(
                self.model_evaluation_config.model_evaluation_dir,
                "metrics.yaml"
            )
            
            with open(metrics_file_path, 'w') as f:
                yaml.dump(metrics, f)
            
            # Determine if model meets acceptance criteria
            is_model_accepted = metrics.get("mAP50", 0) >= self.model_evaluation_config.min_map_threshold
            
            model_evaluation_artifact = ModelEvaluationArtifact(
                is_model_accepted=is_model_accepted,
                evaluated_model_path=self.model_trainer_artifact.trained_model_file_path,
                model_metrics=metrics,
                metrics_file_path=metrics_file_path,
                trained_model_path=self.model_trainer_artifact.trained_model_file_path
            )
            
            logging.info(f"Model evaluation completed. Artifact: {model_evaluation_artifact}")
            return model_evaluation_artifact
            
        except Exception as e:
            logging.error(f"Error in model evaluation: {str(e)}")
            raise AppException(e, sys) 