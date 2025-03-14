#!/usr/bin/env python
"""
Main entry point for running the Wait Time Analysis Streamlit application.
"""
import os
import sys
import subprocess

def main():
    """Run the Streamlit application."""
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Add the project directory to the Python path
    sys.path.insert(0, script_dir)
    
    # Path to the app.py file
    app_path = os.path.join(script_dir, "streamlit_app", "app.py")
    
    # Check if the app.py file exists
    if not os.path.exists(app_path):
        print(f"Error: Could not find {app_path}")
        sys.exit(1)
    
    # Run the Streamlit application
    print("Starting Wait Time Analysis Streamlit application...")
    subprocess.run(["streamlit", "run", app_path])

if __name__ == "__main__":
    main() 