from airflow import DAG
from airflow.utils.dates import days_ago
from airflow.operators.python import PythonOperator
import datetime
from api.utils.get_models import list_model_repository_folders
import os
from dotenv import load_dotenv
from api.utils.resolve_path import resolve_path
from api.utils.make_db import process_listing
from api.utils.predict import predict_from_model_and_df
from mlprojects.production.tf_trimodel_extended import tf_trimodel_extended
from api.utils.metrics import accuracy_from_df
import duckdb


with DAG(
    dag_id="my_very_first_dag",
    description="My first DAG created with DataScientest",
    tags=["tutorial", "datascientest"],
    schedule_interval=None,
    default_args={
        "owner": "airflow",
        "start_date": days_ago(2),
    },
) as my_dag:
    # Définition de la fonction à exécuter
    def print_date_and_hello():
        
        load_dotenv(".envp/.env.airflow")
        print("load_dotenv done")

        duckdb_path = os.path.join(
            resolve_path(os.environ["DATA_PATH"]),
            os.environ["RAKUTEN_DB_NAME"].lstrip("/"),
        )
        
        conn = duckdb.connect(database=duckdb_path, read_only=False)

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
            "imageid"
            ]
        columns_str = ", ".join(cols)
        result = conn.sql(
            f"SELECT {columns_str} FROM fact_listings WHERE validate_datetime >= DATETIME '2024-07-17' AND validate_datetime <= DATETIME '2024-07-18'"
        ).df()
        
        print(result.shape)
        print(result)

        # print(os.environ["AWS_CONFIG_PATH"])
        # print(os.getcwd())
        # print(resolve_path(os.getcwd()))
        # print(os.environ["AWS_CONFIG_PATH"])
        # print(resolve_path(os.environ["AWS_CONFIG_PATH"]))
        # print(list_model_repository_folders('/app/.aws/.encrypted', is_production=True))

        #############################################################################################################

        # listing_df = process_listing(resolve_path('data/X_train.csv'), resolve_path('data/Y_train.csv'))
        # listing_df["image_path"] = listing_df.apply(
        #     lambda row: resolve_path(
        #         f"data/images/image_train/image_{row['imageid']}_product_{row['productid']}.jpg"
        #     ),
        #     axis=1,
        # )

        # to_predict = listing_df.head(10)
        # print(to_predict)

        # model = tf_trimodel_extended("tf_trimodel", "20240708_19-15-54", "production")

        # predicted_df = predict_from_model_and_df(model, to_predict)

        # print(predicted_df)

        # acc = accuracy_from_df(predicted_df, 'tf_trimodel_20240708_19-15-54_production', 'user_prdtypecode')

        # print(acc)
        ##################################################################################################################
        # text_designation = "Jeu video"
        # text_description = "Titi et les bijoux magiques jeux video enfants gameboy advance"
        # image_path = resolve_path("data/images/image_train/image_528113_product_923222.jpg")
        # print(image_path)

        # # Prédiction avec un chemin d'image
        # result = model.predict(text_designation, text_description, image_path)

        # print(result)

    def print_date_and_hello_again():
        print("Hello from Airflow again")

    my_task = PythonOperator(
        task_id="my_very_first_task",
        python_callable=print_date_and_hello,
    )

    my_task2 = PythonOperator(
        task_id="my_second_task",
        python_callable=print_date_and_hello_again,
    )

my_task >> my_task2
