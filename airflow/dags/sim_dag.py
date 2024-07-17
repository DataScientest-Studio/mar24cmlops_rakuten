from airflow import DAG
from datetime import timedelta
from airflow.utils.dates import days_ago
from airflow import DAG
from airflow.utils.dates import days_ago
from airflow.operators.python import PythonOperator
import datetime
import os
from dotenv import load_dotenv
from api.utils.resolve_path import resolve_path
from api.utils.make_db import process_listing
from api.utils.predict import load_models_from_file, predict_from_model_and_df
from mlprojects.production.tf_trimodel_extended import tf_trimodel_extended
from api.utils.metrics import accuracy_from_df
import duckdb

default_args = {
    "owner": "mar24cmlops",
    "retries": 2,
    "retry_delay": timedelta(minutes=2),
}


def retrieve_data_and_predict():
    load_dotenv(".envp/.env.airflow")

    duckdb_path = os.path.join(
        resolve_path(os.environ["DATA_PATH"]),
        os.environ["RAKUTEN_DB_NAME"].lstrip("/"),
    )

    conn = duckdb.connect(database=duckdb_path, read_only=True)

    cols = [
        "designation",
        "description",
        "productid",
        "imageid",
        "user_prdtypecode",
        "model_prdtypecode",
        "waiting_datetime",
        "validate_datetime",
        "status",
        "user",
        "imageid",
    ]
    columns_str = ", ".join(cols)
    result = conn.sql(
        f"SELECT {columns_str} FROM fact_listings WHERE validate_datetime >= DATETIME '2024-07-17' AND validate_datetime <= DATETIME '2024-07-18'"
    ).df()

    result["image_path"] = result.apply(
        lambda row: resolve_path(
            f"data/images/image_train/image_{row['imageid']}_product_{row['productid']}.jpg"
        ),
        axis=1,
    )

    ############
    models = load_models_from_file(os.environ["AWS_CONFIG_PATH"], resolve_path('models/model_list.txt'))
    acc_from_models = {}

    for model in models:
        colname = f"{model.model_name}_{model.version}_{model.model_type}"
        result = predict_from_model_and_df(model, result)
        acc_from_models[colname] = accuracy_from_df(result, colname, "user_prdtypecode")

    print(acc_from_models)


def insert_accuracy_into_db():
    print("accuracy inserted")


with DAG(
    dag_id="sim_dag",
    default_args=default_args,
    description="Simulates a day use of Rakuten Website",
    start_date=days_ago(2),
    schedule_interval=None,
) as dag:
    retrieve_data_and_predict = PythonOperator(
        task_id="retrieve_data_and_predict", python_callable=retrieve_data_and_predict
    )

    insert_accuracy_into_db = PythonOperator(
        task_id="insert_accuracy_into_db", python_callable=insert_accuracy_into_db
    )

retrieve_data_and_predict >> insert_accuracy_into_db
