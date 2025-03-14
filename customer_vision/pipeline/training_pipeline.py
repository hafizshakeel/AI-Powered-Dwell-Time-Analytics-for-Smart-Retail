import sys, os
from customer_vision.logger import logging
from customer_vision.exception import AppException
from customer_vision.components.data_ingestion import DataIngestion
from customer_vision.components.data_validation import DataValidation
from customer_vision.components.model_trainer import ModelTrainer
from customer_vision.components.model_evaluation import ModelEvaluation


from customer_vision.entity.config_entity import (DataIngestionConfig,
                                                 DataValidationConfig,
                                                 ModelTrainerConfig,
                                                 ModelEvaluationConfig)

from customer_vision.entity.artifacts_entity import (DataIngestionArtifact,
                                                    DataValidationArtifact,
                                                    ModelTrainerArtifact,
                                                    ModelEvaluationArtifact)


class TrainPipeline:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        self.data_validation_config = DataValidationConfig()
        self.model_trainer_config = ModelTrainerConfig()
        self.model_evaluation_config = ModelEvaluationConfig()


    
    def start_data_ingestion(self)-> DataIngestionArtifact:
        try: 
            logging.info(
                "Entered the start_data_ingestion method of TrainPipeline class"
            )
            logging.info("Getting the data from URL")

            data_ingestion = DataIngestion(
                data_ingestion_config =  self.data_ingestion_config
            )

            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            logging.info("Got the data from URL")
            logging.info(
                "Exited the start_data_ingestion method of TrainPipeline class"
            )

            return data_ingestion_artifact

        except Exception as e:
            logging.error(f"Error in data ingestion: {str(e)}")
            raise AppException(e, sys)
        


    
    def start_data_validation(
        self, data_ingestion_artifact: DataIngestionArtifact
    ) -> DataValidationArtifact:
        logging.info("Entered the start_data_validation method of TrainPipeline class")

        try:
            data_validation = DataValidation(
                data_ingestion_artifact=data_ingestion_artifact,
                data_validation_config=self.data_validation_config,
            )

            data_validation_artifact = data_validation.initiate_data_validation()

            logging.info("Performed the data validation operation")

            logging.info(
                "Exited the start_data_validation method of TrainPipeline class"
            )

            return data_validation_artifact

        except Exception as e:
            logging.error(f"Error in data validation: {str(e)}")
            raise AppException(e, sys) from e



    
    def start_model_trainer(self
    ) -> ModelTrainerArtifact:
        try:
            logging.info("Entered the start_model_trainer method of TrainPipeline class")
            
            model_trainer = ModelTrainer(
                model_trainer_config=self.model_trainer_config,
            )
            
            model_trainer_artifact = model_trainer.initiate_model_trainer()
            
            logging.info("Performed the model training operation")
            logging.info("Exited the start_model_trainer method of TrainPipeline class")
            
            return model_trainer_artifact

        except Exception as e:
            logging.error(f"Error in model training: {str(e)}")
            raise AppException(e, sys)
    
    
    def start_model_evaluation(self, model_trainer_artifact: ModelTrainerArtifact) -> ModelEvaluationArtifact:
        try:
            logging.info("Entered the start_model_evaluation method of TrainPipeline class")
            
            model_evaluation = ModelEvaluation(
                model_evaluation_config=self.model_evaluation_config,
                model_trainer_artifact=model_trainer_artifact
            )
            
            model_evaluation_artifact = model_evaluation.initiate_model_evaluation()
            
            logging.info(f"Model evaluation completed. Artifact: {model_evaluation_artifact}")
            logging.info("Exited the start_model_evaluation method of TrainPipeline class")
            
            return model_evaluation_artifact
            
        except Exception as e:
            logging.error(f"Error in model evaluation: {str(e)}")
            raise AppException(e, sys)
    

    def run_pipeline(self) -> None:
        try:
            logging.info("Starting the training pipeline")
            
            # Step 1: Data Ingestion
            logging.info("Step 1: Data Ingestion")
            data_ingestion_artifact = self.start_data_ingestion()
            
            # Step 2: Data Validation
            logging.info("Step 2: Data Validation")
            data_validation_artifact = self.start_data_validation(
                data_ingestion_artifact=data_ingestion_artifact
            )

            if data_validation_artifact.validation_status:
                # Step 3: Model Training
                logging.info("Step 3: Model Training")
                model_trainer_artifact = self.start_model_trainer()
                
                # Step 4: Model Evaluation
                logging.info("Step 4: Model Evaluation")
                model_evaluation_artifact = self.start_model_evaluation(
                    model_trainer_artifact=model_trainer_artifact
                )
                
                if model_evaluation_artifact.is_model_accepted:
                    logging.info("Model accepted and ready for deployment")
                else:
                    logging.info("Model rejected as it did not meet performance criteria")
                    
                logging.info("Training pipeline completed successfully")
                return model_evaluation_artifact
            
            else:
                logging.error("Data validation failed. Cannot proceed with model training.")
                raise Exception("Data validation failed. Your data is not in the correct format.")

        except Exception as e:
            logging.error(f"Error in training pipeline: {str(e)}")
            raise AppException(e, sys)


