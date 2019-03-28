import os
import logging
from dotenv import load_dotenv
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
load_dotenv(verbose=True, dotenv_path=os.path.join(PROJECT_ROOT, '.env'))
SOLVERS_DIR = os.path.join(PROJECT_ROOT, "solvers")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
KINDDB_SERVICE_URL = os.getenv('KINDDB_KSVC_ENDPOINT_URL', 'http://localhost:8008/graphql')
SERVICE_ID = os.getenv('SERVICE_ID')
SERVICE_ADDRESS = '0.0.0.0'
SERVICE_PORT = os.getenv('PORT', '7357')
RABBITMQ_ADDR = os.getenv('RABBITMQ_ADDR')
RABBITMQ_PORT = os.getenv('RABBITMQ_PORT')
CKG_SERVICE_URL = os.getenv('CKG_SERVICE_URL', 'http://localhost:8011/service/io.maana.metalearning/graphql')
REMOTE_KSVC_ENDPOINT_URL = os.getenv('REMOTE_KSVC_ENDPOINT_URL', 'http://localhost:8003/graphql')
LOG_LEVEL = logging.DEBUG