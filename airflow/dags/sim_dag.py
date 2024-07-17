from airflow import DAG
from datetime import timedelta, datetime
from airflow.utils.dates import days_ago
from airflow.operators.python import PythonOperator
import os
import json
from dotenv import load_dotenv
from api.utils.resolve_path import resolve_path
from api.utils.make_db import process_listing
from api.utils.predict import load_models_from_file, predict_from_model_and_df
from mlprojects.production.tf_trimodel_extended import tf_trimodel_extended
from api.utils.metrics import accuracy_from_df
import duckdb
from airflow.sensors.external_task import ExternalTaskMarker

# Define default arguments for the DAG
default_args = {
    "owner": "mar24cmlops",
    "retries": 2,
    "retry_delay": timedelta(minutes=2),
}

# Path to the file storing the last successful date
last_successful_date_path = "/app/data/last_successful_date.json"


def read_last_successful_date():
    """
    Reads the last successful date from a file. If the file does not exist,
    returns the default start date.

    :return: The last successful date or the default start date
    """
    if os.path.exists(last_successful_date_path):
        with open(last_successful_date_path, "r") as f:
            data = json.load(f)
            return datetime.strptime(data["last_successful_date"], "%Y-%m-%d")
    else:
        return datetime(2024, 7, 17)


def write_last_successful_date(date):
    """
    Writes the last successful date to a file.

    :param date: The last successful date to be written
    """
    with open(last_successful_date_path, "w") as f:
        json.dump({"last_successful_date": date.strftime("%Y-%m-%d")}, f)


def retrieve_data_and_predict(**kwargs):
    """
    Retrieves data for the current date, makes predictions using models,
    calculates the accuracy of the models, and inserts the results into DuckDB.

    :param kwargs: context arguments provided by Airflow
    """
    load_dotenv(".envp/.env.airflow")

    # Define the path to the DuckDB database
    duckdb_path = os.path.join(
        resolve_path(os.environ["DATA_PATH"]),
        os.environ["RAKUTEN_DB_NAME"].lstrip("/"),
    )

    # Connect to the DuckDB database
    conn = duckdb.connect(database=duckdb_path, read_only=False)

    # Get the last successful date
    last_successful_date = read_last_successful_date()

    # Columns to select from the fact_listings table
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

    # Calculate the next day's date
    next_date = last_successful_date + timedelta(days=1)

    # Retrieve data for the current date
    result = conn.sql(
        f"SELECT {columns_str} FROM fact_listings WHERE validate_datetime >= DATETIME '{last_successful_date.strftime('%Y-%m-%d')}' AND validate_datetime < DATETIME '{next_date.strftime('%Y-%m-%d')}'"
    ).df()

    # Add image paths to the dataframe
    result["image_path"] = result.apply(
        lambda row: resolve_path(
            f"data/images/image_train/image_{row['imageid']}_product_{row['productid']}.jpg"
        ),
        axis=1,
    )

    # Load models
    models = load_models_from_file(
        os.environ["AWS_CONFIG_PATH"], resolve_path("models/model_list.txt")
    )
    acc_from_models = {}

    # Make predictions and calculate accuracy for each model
    for model in models:
        colname = f"{model.model_name}_{model.version}_{model.model_type}"
        result = predict_from_model_and_df(model, result)
        acc_from_models[colname] = accuracy_from_df(result, colname, "user_prdtypecode")

    print(acc_from_models)

    # Prepare the columns for table creation and data insertion
    columns_definitions = ", ".join(
        [f'"{model_name}" DOUBLE' for model_name in acc_from_models.keys()]
    )
    columns_names = ", ".join(
        [f'"{model_name}"' for model_name in acc_from_models.keys()]
    )
    columns_values = ", ".join(
        str(acc_from_models[model_name]) for model_name in acc_from_models.keys()
    )

    # Create the accuracy_daily table if it does not exist
    create_table_query = f"""
    CREATE TABLE IF NOT EXISTS accuracy_daily (
        date DATE,
        {columns_definitions}
    )
    """
    conn.execute(create_table_query)

    # Insert accuracy results into the accuracy_daily table
    insert_query = f"""
    INSERT INTO accuracy_daily (date, {columns_names})
    VALUES (
        '{last_successful_date.strftime('%Y-%m-%d')}',
        {columns_values}
    )
    """
    conn.execute(insert_query)

    # Update the last successful date for the next run
    write_last_successful_date(next_date)


def compute_rolling_metrics():
    """
    Computes rolling metrics (5-day moving average, standard deviation, and variance)
    for each model's accuracy and inserts the results into the rolling_metrics table.
    """
    load_dotenv(".envp/.env.airflow")

    # Define the path to the DuckDB database
    duckdb_path = os.path.join(
        resolve_path(os.environ["DATA_PATH"]),
        os.environ["RAKUTEN_DB_NAME"].lstrip("/"),
    )

    # Connect to the DuckDB database
    conn = duckdb.connect(database=duckdb_path, read_only=True)

    # Retrieve the accuracy data from the last 5 days
    accuracy_df = conn.sql(
        "SELECT * FROM accuracy_daily ORDER BY date DESC LIMIT 5"
    ).df()
    conn.close()
    if accuracy_df.empty:
        print("No data found in accuracy_daily table.")
        return

    # Compute rolling metrics
    rolling_metrics_df = (
        accuracy_df.set_index("date")
        .sort_index()
        .rolling(window=5)
        .agg(["mean", "std", "var"])
        .dropna()
    )

    # Flatten the multi-level columns
    rolling_metrics_df.columns = [
        "_".join(col).strip() for col in rolling_metrics_df.columns.values
    ]

    # Reset index to include 'date' as a column
    rolling_metrics_df.reset_index(inplace=True)

    # Prepare the data for insertion
    data_to_insert = []

    for _, row in rolling_metrics_df.iterrows():
        date = row["date"]
        for model_name in accuracy_df.columns[1:]:  # Skip the 'date' column
            rolling_mean = row[f"{model_name}_mean"]
            rolling_std = row[f"{model_name}_std"]
            rolling_var = row[f"{model_name}_var"]
            data_to_insert.append(
                (date, model_name, rolling_mean, rolling_std, rolling_var)
            )

    # Insert rolling metrics into the rolling_metrics table

    conn = duckdb.connect(database=duckdb_path, read_only=False)

    # Create the rolling_metrics table if it does not exist
    create_table_query = """
    CREATE TABLE IF NOT EXISTS rolling_metrics (
        date DATE,
        model_name VARCHAR,
        rolling_mean DOUBLE,
        rolling_std DOUBLE,
        rolling_var DOUBLE,
        PRIMARY KEY (date, model_name)
    )
    """
    conn.execute(create_table_query)
    conn.close()

    # Prepare and execute the insertion query

    insert_query = """
    INSERT INTO rolling_metrics (date, model_name, rolling_mean, rolling_std, rolling_var)
    VALUES 
    """
    insert_query += ", ".join(
        f"('{date}', '{model}', {rolling_mean}, {rolling_std}, {rolling_var})"
        for date, model, rolling_mean, rolling_std, rolling_var in data_to_insert
    )
    conn = duckdb.connect(database=duckdb_path, read_only=False)
    conn.execute(insert_query)

    conn.close()

    print("Rolling Metrics computed")


# Define the DAG with arguments and schedule interval
with DAG(
    dag_id="sim_dag",
    default_args=default_args,
    description="Simulates a day use of Rakuten Website",
    start_date=days_ago(2),
    schedule_interval="*/10 * * * *",  # Run every 10 minutes
    catchup=False,
    tags=["marc24mlops"],
) as dag:
    # Define task to retrieve data and make predictions
    retrieve_data_and_predict = PythonOperator(
        task_id="retrieve_data_and_predict",
        python_callable=retrieve_data_and_predict,
        provide_context=True,
    )

    # Define task to insert accuracy data into the database
    compute_rolling_metrics = PythonOperator(
        task_id="compute_rolling_metrics", python_callable=compute_rolling_metrics
    )

    end_task = ExternalTaskMarker(
        task_id="end_task",
        external_dag_id="model_evaluate_and_retrain_dag",
        external_task_id=None,
    )
# Set the task dependencies
retrieve_data_and_predict >> compute_rolling_metrics >> end_task
