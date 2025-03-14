import os
import sys
import yaml
import shutil
from datetime import datetime
from ultralytics import YOLO
from customer_vision.utils.main_utils import read_yaml_file
from customer_vision.logger import logging
from customer_vision.exception import AppException
from customer_vision.entity.config_entity import ModelTrainerConfig
from customer_vision.entity.artifacts_entity import ModelTrainerArtifact


class ModelTrainer:
    def __init__(
        self,
        model_trainer_config: ModelTrainerConfig,
    ):
        self.model_trainer_config = model_trainer_config
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def save_training_artifacts(self, results_dir: str, save_dir: str):
        """
        Save essential training artifacts
        """
        try:
            # Create plots directory
            plots_dir = os.path.join(save_dir, "plots")
            os.makedirs(plots_dir, exist_ok=True)

            # Important plots to save
            plot_files = [
                "results.png",
                "confusion_matrix.png",
                "labels.jpg",
                "val_batch0_pred.jpg"
            ]
            
            for plot_file in plot_files:
                src = os.path.join(results_dir, plot_file)
                if os.path.exists(src):
                    dst = os.path.join(plots_dir, plot_file)
                    shutil.copy2(src, dst)
                    logging.info(f"Saved {plot_file}")
                    
            # Save training metrics
            results_csv = os.path.join(results_dir, "results.csv")
            if os.path.exists(results_csv):
                dst = os.path.join(save_dir, "results.csv")
                shutil.copy2(results_csv, dst)
                logging.info(f"Saved training results to {dst}")
                
        except Exception as e:
            logging.warning(f"Error saving training artifacts: {str(e)}")

    def initiate_model_trainer(self,) -> ModelTrainerArtifact:
        logging.info("Entered initiate_model_trainer method of ModelTrainer class")

        try:
            # Create model trainer directory
            os.makedirs(self.model_trainer_config.model_trainer_dir, exist_ok=True)
            
            # Get the path to data.yaml in the feature store
            feature_store_path = os.path.join("artifacts", "data_ingestion", "feature_store")
            data_yaml_path = os.path.join(feature_store_path, "data.yaml")
            
            logging.info(f"Reading data configuration from {data_yaml_path}")
            
            # Load and update data.yaml with absolute paths
            with open(data_yaml_path, 'r') as stream:
                data_config = yaml.safe_load(stream)
                num_classes = data_config['nc']
                class_names = data_config['names']
            
            # Create a copy of data.yaml with absolute paths for training
            training_yaml_path = os.path.join(self.model_trainer_config.model_trainer_dir, "data.yaml")
            
            # Update paths to be absolute
            data_config['train'] = os.path.abspath(os.path.join(feature_store_path, "train", "images"))
            data_config['val'] = os.path.abspath(os.path.join(feature_store_path, "valid", "images"))
            data_config['test'] = os.path.abspath(os.path.join(feature_store_path, "test", "images"))
            
            # Write the updated config to the training directory
            with open(training_yaml_path, 'w') as f:
                yaml.dump(data_config, f)
            
            logging.info(f"Updated data.yaml created at {training_yaml_path}")
            logging.info(f"Training model for {num_classes} classes: {class_names}")
            
            # Initialize YOLOv8 model with pretrained weights
            model = YOLO(self.model_trainer_config.weight_name)
            
            # Train the model with the updated data.yaml
            logging.info(f"Starting model training with {self.model_trainer_config.no_epochs} epochs")
            results = model.train(
                data=training_yaml_path,
                epochs=self.model_trainer_config.no_epochs,
                batch=self.model_trainer_config.batch_size,
                imgsz=self.model_trainer_config.image_size,
                name="train",
                exist_ok=True,
                verbose=True,
                plots=True
            )
            
            # Get the results directory
            results_dir = os.path.join("runs", "detect", "train")
            
            # Save important training artifacts
            self.save_training_artifacts(results_dir, self.model_trainer_config.model_trainer_dir)
            
            # Copy best model to model trainer directory
            best_model_path = os.path.join(results_dir, "weights", "best.pt")
            trained_model_path = os.path.join(self.model_trainer_config.model_trainer_dir, "best.pt")
            
            if os.path.exists(best_model_path):
                shutil.copy2(best_model_path, trained_model_path)
                logging.info(f"Best model saved to {trained_model_path}")
            else:
                logging.warning(f"Best model not found at {best_model_path}")
            
            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=trained_model_path
            )

            logging.info("Model training completed successfully")
            logging.info(f"Model trainer artifact: {model_trainer_artifact}")

            return model_trainer_artifact

        except Exception as e:
            raise AppException(e, sys)




            