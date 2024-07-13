from api.utils.get_models import verify_and_download_models
from api.utils.predict import load_models_from_file
from api.utils.make_db import process_listing
from api.utils.resolve_path import resolve_path
import pandas as pd
from dotenv import load_dotenv
import os
from datetime import datetime,timedelta






if __name__=="__main__":
    global aws_config_path, duckdb_path, encrypted_file_path, conn, mdl_list, s3_conn

    # Load environment variables from .env file
    env_path = resolve_path(".envp/.env.development")
    load_dotenv(env_path)

    # Convert paths to absolute paths
    aws_config_path = resolve_path(os.environ["AWS_CONFIG_PATH"])
    duckdb_path = os.path.join(
        resolve_path(os.environ["DATA_PATH"]), os.environ["RAKUTEN_DB_NAME"].lstrip("/")
    )
    encrypted_file_path = os.path.join(
        resolve_path(os.environ["AWS_CONFIG_FOLDER"]), ".encrypted"
    )

    # Verify and download models if necessary
    verify_and_download_models(aws_config_path, resolve_path("models/model_list.txt"))
    
    # Load model list
#    mdl_list = load_models_from_file(
#        aws_config_path, resolve_path("models/model_list.txt")
#    )
    
    X_pathway=resolve_path(os.path.join('data','X_train.csv'))
    Y_pathway=resolve_path(os.path.join('data','Y_train.csv'))
    df=process_listing(X_pathway,Y_pathway)
    print(df.head())
    longueur=df.shape[0]
    init_ind=int(longueur*3/10)
    init_date=datetime.strptime("2024-07-01","%Y-%m-%d")
    print(init_date)
    for i in range(init_ind,(longueur-init_ind),100):
        df.loc[(init_ind+i):(init_ind+i+99),'validate_datetime']=init_date+timedelta(days=int(i/100))
### Prediction sur les 30% 