import os
import sys
import zipfile
import requests
import shutil
from tqdm import tqdm
from pathlib import Path
from customer_vision.logger import logging
from customer_vision.exception import AppException
from customer_vision.entity.config_entity import DataIngestionConfig
from customer_vision.entity.artifacts_entity import DataIngestionArtifact


class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig = DataIngestionConfig()):
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
           raise AppException(e, sys)

    def verify_dataset_structure(self, feature_store_path: str) -> bool:
        """
        Verify the extracted dataset has the correct structure for YOLOv8 training
        """
        try:
            required_dirs = ['train/images', 'train/labels', 'valid/images', 'valid/labels']
            yaml_file = Path(feature_store_path) / 'data.yaml'
            
            # Check if data.yaml exists and is valid
            if not yaml_file.exists():
                logging.error("data.yaml not found in the dataset")
                return False
            
            try:
                import yaml
                with open(yaml_file, 'r') as f:
                    yaml_content = yaml.safe_load(f)
                    required_fields = ['train', 'val', 'nc', 'names']
                    for field in required_fields:
                        if field not in yaml_content:
                            logging.error(f"Required field '{field}' not found in data.yaml")
                            return False
                    # Verify it's configured for customer detection
                    if yaml_content.get('nc') != 1 or 'customer' not in [name.lower() for name in yaml_content.get('names', [])]:
                        logging.error("data.yaml is not properly configured for customer detection")
                        return False
            except Exception as e:
                logging.error(f"Error reading data.yaml: {str(e)}")
                return False
                
            # Check all required directories exist and have files
            for dir_path in required_dirs:
                full_path = Path(feature_store_path) / dir_path
                if not full_path.exists():
                    logging.error(f"Required directory {dir_path} not found")
                    return False
                files = list(full_path.glob('*'))
                if not files:
                    logging.error(f"No files found in {dir_path}")
                    return False
                logging.info(f"Found {len(files)} files in {dir_path}")
                    
            logging.info("Dataset structure verification completed successfully")
            return True
            
        except Exception as e:
            logging.error(f"Error during dataset verification: {str(e)}")
            return False

    def check_existing_dataset(self) -> bool:
        """
        Check if dataset already exists and is valid
        """
        try:
            feature_store_path = self.data_ingestion_config.feature_store_file_path
            if not os.path.exists(feature_store_path):
                logging.info("Dataset not found, will proceed with download")
                return False
                
            logging.info("Found existing dataset, verifying structure...")
            if self.verify_dataset_structure(feature_store_path):
                logging.info("Existing dataset is valid, skipping download")
                return True
            else:
                logging.warning("Existing dataset is invalid, will re-download")
                return False
                
        except Exception as e:
            logging.error(f"Error checking existing dataset: {str(e)}")
            return False

    def download_data(self)-> str:
        '''
        Fetch data from the Roboflow URL with progress bar
        '''
        try: 
            dataset_url = self.data_ingestion_config.data_download_url
            zip_download_dir = self.data_ingestion_config.data_ingestion_dir
            os.makedirs(zip_download_dir, exist_ok=True)
            data_file_name = "data.zip"
            zip_file_path = os.path.join(zip_download_dir, data_file_name)
            
            # Clean up any existing download
            if os.path.exists(zip_file_path):
                os.remove(zip_file_path)
            
            logging.info(f"Initiating download from Roboflow")

            # Stream download with progress bar
            response = requests.get(dataset_url, stream=True)
            if response.status_code != 200:
                raise Exception(f"Failed to download data from Roboflow. Status code: {response.status_code}")
            
            total_size = int(response.headers.get('content-length', 0))
            block_size = 1024  # 1 KB

            with open(zip_file_path, 'wb') as f, tqdm(
                desc="Downloading dataset",
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for data in response.iter_content(block_size):
                    size = f.write(data)
                    pbar.update(size)

            if os.path.getsize(zip_file_path) == 0:
                raise Exception("Downloaded file is empty")

            logging.info(f"Successfully downloaded dataset to {zip_file_path}")
            return zip_file_path

        except Exception as e:
            if os.path.exists(zip_file_path):
                os.remove(zip_file_path)
            raise AppException(e, sys)

    def extract_zip_file(self,zip_file_path: str)-> str:
        """
        Extracts the zip file with progress bar
        """
        try:
            feature_store_path = self.data_ingestion_config.feature_store_file_path
            
            # Clean up existing feature store if it exists
            if os.path.exists(feature_store_path):
                shutil.rmtree(feature_store_path)
            
            os.makedirs(feature_store_path, exist_ok=True)
            
            with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                # Get total size for progress bar
                total_size = sum(file.file_size for file in zip_ref.filelist)
                extracted_size = 0
                
                with tqdm(
                    desc="Extracting dataset",
                    total=total_size,
                    unit='B',
                    unit_scale=True,
                    unit_divisor=1024,
                ) as pbar:
                    for file in zip_ref.filelist:
                        zip_ref.extract(file, feature_store_path)
                        extracted_size += file.file_size
                        pbar.update(file.file_size)

            logging.info(f"Dataset extracted successfully to {feature_store_path}")
            return feature_store_path

        except Exception as e:
            if os.path.exists(feature_store_path):
                shutil.rmtree(feature_store_path)
            raise AppException(e, sys)

    def initiate_data_ingestion(self)-> DataIngestionArtifact:
        logging.info("Starting data ingestion process")
        try: 
            # First check if valid dataset already exists
            if self.check_existing_dataset():
                feature_store_path = self.data_ingestion_config.feature_store_file_path
                # Return artifact without downloading
                return DataIngestionArtifact(
                    data_zip_file_path="",  # Empty since we didn't download
                    feature_store_path=feature_store_path
                )
            
            # If not exists or invalid, proceed with download
            zip_file_path = self.download_data()
            feature_store_path = self.extract_zip_file(zip_file_path)
            
            # Verify dataset structure
            if not self.verify_dataset_structure(feature_store_path):
                raise Exception("Dataset verification failed. Please check the logs for details.")
            
            data_ingestion_artifact = DataIngestionArtifact(
                data_zip_file_path=zip_file_path,
                feature_store_path=feature_store_path
            )

            logging.info("Data ingestion completed successfully")
            logging.info(f"Data ingestion artifact: {data_ingestion_artifact}")

            return data_ingestion_artifact

        except Exception as e:
            logging.error("Data ingestion failed")
            raise AppException(e, sys)

