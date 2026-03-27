import os


DATA_PATH = os.path.join(os.path.dirname(__file__), 'flood_data.csv')


MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')
MODEL_FILE = os.path.join(MODELS_DIR, 'models.pkl')


TEST_SIZE = 0.2
RANDOM_STATE = 42


DEBUG = False  