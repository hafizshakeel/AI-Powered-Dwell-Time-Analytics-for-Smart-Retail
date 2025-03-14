import os,sys
import shutil
import yaml
from customer_vision.logger import logging
from customer_vision.exception import AppException
from customer_vision.entity.config_entity import DataValidationConfig
from customer_vision.entity.artifacts_entity import (DataIngestionArtifact,
                                                 DataValidationArtifact)


class DataValidation:
    def __init__(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        data_validation_config: DataValidationConfig,
    ):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config

        except Exception as e:
            raise AppException(e, sys) 
        
    def validate_all_files_exist(self)-> bool:
        try:
            validation_status = True
            all_files = os.listdir(self.data_ingestion_artifact.feature_store_path)
            
            for file in self.data_validation_config.required_file_list:
                if file not in all_files:
                    validation_status = False
                    logging.error(f"Required file {file} not found in feature store")
            
            # Check if train, valid, and test directories exist
            required_dirs = ["train", "valid", "test"]
            for dir_name in required_dirs:
                dir_path = os.path.join(self.data_ingestion_artifact.feature_store_path, dir_name)
                if not os.path.exists(dir_path):
                    validation_status = False
                    logging.error(f"Required directory {dir_name} not found in feature store")
                else:
                    # Check if images directory exists within each directory
                    images_dir = os.path.join(dir_path, "images")
                    if not os.path.exists(images_dir):
                        validation_status = False
                        logging.error(f"Images directory not found in {dir_name}")
                    elif len(os.listdir(images_dir)) == 0:
                        validation_status = False
                        logging.error(f"No images found in {dir_name}/images")
                    else:
                        logging.info(f"Found {len(os.listdir(images_dir))} images in {dir_name}/images")
            
            # Create validation directory and write status
            os.makedirs(self.data_validation_config.data_validation_dir, exist_ok=True)
            with open(self.data_validation_config.valid_status_file_dir, 'w') as f:
                f.write(f"Validation status: {validation_status}")
                
            return validation_status

        except Exception as e:
            raise AppException(e, sys)
    
    def validate_yaml_file(self) -> bool:
        """
        Validate the data.yaml file to ensure it has the required fields
        and fix paths if necessary
        """
        try:
            yaml_path = os.path.join(self.data_ingestion_artifact.feature_store_path, "data.yaml")
            if not os.path.exists(yaml_path):
                logging.error("data.yaml file not found")
                return False
                
            with open(yaml_path, 'r') as f:
                data = yaml.safe_load(f)
                
            # Check required fields
            required_fields = ['train', 'val', 'nc', 'names']
            for field in required_fields:
                if field not in data:
                    logging.error(f"Required field '{field}' not found in data.yaml")
                    return False
            
            # Validate and fix paths if needed
            modified = False
            
            # Check if train path exists
            train_path = data['train']
            if not os.path.exists(os.path.join(self.data_ingestion_artifact.feature_store_path, train_path.lstrip('../'))):
                # Try to fix the path
                corrected_path = "../train/images"
                logging.warning(f"Train path '{train_path}' not found. Setting to '{corrected_path}'")
                data['train'] = corrected_path
                modified = True
                
            # Check if val path exists
            val_path = data['val']
            if not os.path.exists(os.path.join(self.data_ingestion_artifact.feature_store_path, val_path.lstrip('../'))):
                # Try to fix the path
                corrected_path = "../valid/images"
                logging.warning(f"Val path '{val_path}' not found. Setting to '{corrected_path}'")
                data['val'] = corrected_path
                modified = True
                
            # Check if test path exists (if present)
            if 'test' in data:
                test_path = data['test']
                if not os.path.exists(os.path.join(self.data_ingestion_artifact.feature_store_path, test_path.lstrip('../'))):
                    # Try to fix the path
                    corrected_path = "../test/images"
                    logging.warning(f"Test path '{test_path}' not found. Setting to '{corrected_path}'")
                    data['test'] = corrected_path
                    modified = True
            else:
                # Add test path if missing
                data['test'] = "../test/images"
                modified = True
                
            # Check if classes match expected detection classes
            if 'names' in data:
                logging.info(f"Classes in data.yaml: {data['names']}")
                # Update class names for customer detection if needed
                if len(data['names']) != 1 or 'customer' not in [name.lower() for name in data['names']]:
                    logging.warning("Updating class names for customer detection")
                    data['names'] = ['customer']
                    data['nc'] = 1
                    modified = True
            
            # Write back the modified yaml if changes were made
            if modified:
                logging.info("Updating data.yaml with corrected paths and class names")
                with open(yaml_path, 'w') as f:
                    yaml.dump(data, f)
                
            return True
                
        except Exception as e:
            logging.error(f"Error validating data.yaml: {str(e)}")
            return False
        
    def initiate_data_validation(self) -> DataValidationArtifact: 
        logging.info("Entered initiate_data_validation method of DataValidation class")
        try:
            files_status = self.validate_all_files_exist()
            yaml_status = self.validate_yaml_file()
            
            validation_status = files_status and yaml_status
            
            data_validation_artifact = DataValidationArtifact(
                validation_status=validation_status)

            logging.info("Exited initiate_data_validation method of DataValidation class")
            logging.info(f"Data validation artifact: {data_validation_artifact}")

            if validation_status:
                logging.info("Data validation successful")
            else:
                logging.error("Data validation failed")

            return data_validation_artifact

        except Exception as e:
            raise AppException(e, sys)
        
