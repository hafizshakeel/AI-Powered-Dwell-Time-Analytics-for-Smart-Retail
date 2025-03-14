import os
import sys
import traceback
from customer_vision.pipeline.training_pipeline import TrainPipeline
from customer_vision.logger import logging
from customer_vision.exception import AppException

def main():
    try:
        logging.info("Starting the training pipeline")
        
        # Check if artifacts directory exists
        if not os.path.exists("artifacts"):
            logging.warning("Artifacts directory not found. Creating it.")
            os.makedirs("artifacts", exist_ok=True)
        
        # Check if data.yaml exists in the expected location
        data_yaml_path = os.path.join("artifacts", "data_ingestion", "feature_store", "data.yaml")
        if os.path.exists(data_yaml_path):
            logging.info(f"Found data.yaml at {data_yaml_path}")
        else:
            logging.warning(f"data.yaml not found at {data_yaml_path}. Will be created during data ingestion.")
        
        # Initialize and run the pipeline
        pipeline = TrainPipeline()
        result = pipeline.run_pipeline()
        
        logging.info("Training pipeline completed successfully")
        if result and hasattr(result, 'is_model_accepted') and result.is_model_accepted:
            logging.info(f"Model accepted with metrics: {result.model_metrics}")
            logging.info(f"Trained model saved at: {result.trained_model_path}")
        
        return result
        
    except Exception as e:
        logging.error("=" * 80)
        logging.error("Training pipeline failed with the following error:")
        logging.error(str(e))
        logging.error("-" * 80)
        logging.error("Stack trace:")
        logging.error(traceback.format_exc())
        logging.error("=" * 80)
        
        # Helpful information for common errors
        error_str = str(e).lower()
        if "data.yaml" in error_str and "not found" in error_str:
            logging.error("SOLUTION: The data.yaml file is missing or has incorrect paths.")
            logging.error("1. Check if the data was downloaded correctly during data ingestion.")
            logging.error("2. Verify that the train, valid, and test directories exist with images.")
            logging.error("3. Run the pipeline again to attempt automatic correction.")
        elif "cuda" in error_str or "gpu" in error_str:
            logging.error("SOLUTION: There might be an issue with GPU acceleration.")
            logging.error("1. Try running with CPU only by setting device='cpu' in your configuration.")
        
        raise AppException(e, sys)

if __name__ == "__main__":
    try:
        main()
        print("\n" + "="*50)
        print("Training completed successfully!")
        print("="*50 + "\n")
    except Exception as e:
        print("\n" + "="*50)
        print("Training failed. See logs for details.")
        print("="*50 + "\n")
        sys.exit(1) 