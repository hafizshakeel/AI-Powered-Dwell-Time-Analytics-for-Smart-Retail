import os
from pathlib import Path
import logging
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_project_structure(project_name: str):
    """
    Create a complete Retail Dwell Time Analysis project structure.
    
    Args:
        project_name (str): Base name of the project
    """
    # Define module names based on project name
    calculate_zone_time = f"{project_name}_zone_time"
    customer_vision = f"{project_name}_vision"
    streamlit_app = "streamlit_app"
    
    # Core module structure (calculate_zone_time)
    core_files = [
        # Main module files
        f'{calculate_zone_time}/__init__.py',
        f'{calculate_zone_time}/__main__.py',
        f'{calculate_zone_time}/setup.py',
        f'{calculate_zone_time}/README.md',
        
        # Core components
        f'{calculate_zone_time}/core/__init__.py',
        f'{calculate_zone_time}/core/detector.py',
        f'{calculate_zone_time}/core/tracker.py',
        
        # Visualization
        f'{calculate_zone_time}/visualization/__init__.py',
        f'{calculate_zone_time}/visualization/annotator.py',
        
        # Configuration
        f'{calculate_zone_time}/config/__init__.py',
        f'{calculate_zone_time}/config/settings.py',
        
        # Utils
        f'{calculate_zone_time}/utils/__init__.py',
        f'{calculate_zone_time}/utils/cli.py',
        f'{calculate_zone_time}/utils/general.py',
        f'{calculate_zone_time}/utils/video.py',
        f'{calculate_zone_time}/utils/timers.py',
        
        # Scripts
        f'{calculate_zone_time}/scripts/__init__.py',
        f'{calculate_zone_time}/scripts/draw_zones_cli.py',
        f'{calculate_zone_time}/scripts/draw_zones_ui.py',
        f'{calculate_zone_time}/scripts/stream_from_file.py',
    ]

    # Training pipeline structure (customer_vision)
    training_files = [
        # Main module files
        f'{customer_vision}/__init__.py',
        f'{customer_vision}/README.md',
        
        # Components
        f'{customer_vision}/components/__init__.py',
        f'{customer_vision}/components/data_ingestion.py',
        f'{customer_vision}/components/data_validation.py',
        f'{customer_vision}/components/model_trainer.py',
        f'{customer_vision}/components/model_evaluation.py',
        
        # Constants
        f'{customer_vision}/constants/__init__.py',
        f'{customer_vision}/constants/training_pipeline/__init__.py',
        f'{customer_vision}/constants/application.py',
        
        # Entity
        f'{customer_vision}/entity/__init__.py',
        f'{customer_vision}/entity/config_entity.py',
        f'{customer_vision}/entity/artifacts_entity.py',
        
        # Exception and logging
        f'{customer_vision}/exception/__init__.py',
        f'{customer_vision}/logger/__init__.py',
        
        # Pipeline
        f'{customer_vision}/pipeline/__init__.py',
        f'{customer_vision}/pipeline/training_pipeline.py',
        
        # Utils
        f'{customer_vision}/utils/__init__.py',
        f'{customer_vision}/utils/main_utils.py',
    ]

    # Streamlit app structure
    streamlit_files = [
        # Main app files
        f'{streamlit_app}/app.py',
        f'{streamlit_app}/README.md',
        f'{streamlit_app}/uploaded_zones.json',
        
        # Processing
        f'{streamlit_app}/processing/__init__.py',
        f'{streamlit_app}/processing/video_processor.py',
        f'{streamlit_app}/processing/rtsp_processor.py',
        
        # Analytics
        f'{streamlit_app}/analytics/__init__.py',
        f'{streamlit_app}/analytics/dashboard.py',
        
        # Alerts
        f'{streamlit_app}/alerts/__init__.py',
        f'{streamlit_app}/alerts/alert_system.py',
        
        # Utils
        f'{streamlit_app}/utils/__init__.py',
        f'{streamlit_app}/utils/file_utils.py',
        f'{streamlit_app}/utils/zone_utils.py',
    ]

    # Root project files
    root_files = [
        # Main files
        'README.md',
        'requirements.txt',
        'setup.py',
        'run_streamlit.py',
        'run_cli.py',
        'train.py',
        
        # Docker and CI/CD
        'Dockerfile',
        '.github/workflows/ci-cd.yml',
        
        # Data directories
        'data/.gitkeep',
        'weights/.gitkeep',
        'artifacts/.gitkeep',
    ]

    # Combine all files
    all_files = core_files + training_files + streamlit_files + root_files

    # Create files and directories
    for file_path in all_files:
        filepath = Path(file_path)
        filedir, filename = os.path.split(filepath)

        if filedir:
            os.makedirs(filedir, exist_ok=True)
            logging.info(f'Created directory: {filedir}')

        if not os.path.exists(filepath):
            with open(filepath, 'w') as f:
                pass
            logging.info(f'Created file: {filepath}')
        else:
            logging.info(f'File already exists: {filepath}')

def main():
    parser = argparse.ArgumentParser(description='Create Retail Dwell Time Analysis project structure')
    parser.add_argument('--name', type=str, required=True, help='Base name for the project (e.g., retail_analytics)')
    args = parser.parse_args()

    project_name = args.name.lower().replace(' ', '_')
    
    print(f"\nCreating Retail Dwell Time Analysis project: {project_name}")
    create_project_structure(project_name)
    
    print("\nProject structure created successfully!")
    print("\nNext steps:")
    print("1. Update setup.py with project dependencies")
    print("2. Configure training pipeline in training_pipeline.py")
    print("3. Implement core application components")
    print("4. Set up Docker and CI/CD if needed")

if __name__ == "__main__":
    main()