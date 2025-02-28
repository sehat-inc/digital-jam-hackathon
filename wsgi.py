import sys
import os

# Add the project directory to the Python path
project_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_dir)

# Import your Flask application
from src.app import app as application

# This allows the application to be run with a WSGI server
if __name__ == "__main__":
    application.run()