import utils.make_db
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv('.env/.env.development')
        
aws_config_path = os.environ['AWS_CONFIG_PATH']
duckdb_path = os.path.join(os.environ['DATA_PATH'], os.environ['RAKUTEN_DB_NAME'].lstrip('/'))
rakuten_db_name = os.environ['RAKUTEN_DB_NAME']

utils.make_db.init_db('data/rakuten_db.duckdb')
