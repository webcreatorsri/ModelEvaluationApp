import sys
import os

# Add the app directory to the path
path = 'C:\Users\HP\Desktop\ai-model\app.py'
if path not in sys.path:
    sys.path.append(path)

# Set the environment variable for the Flask app
os.environ['FLASK_APP'] = 'app.py'

# Import the app object
from app import app as application
