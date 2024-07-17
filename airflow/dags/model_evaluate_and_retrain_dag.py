from airflow import DAG
from airflow.utils.dates import days_ago
from airflow.operators.python import PythonOperator
import os
from dotenv import load_dotenv
from api.utils.resolve_path import resolve_path
from api.utils.predict import predict_from_model_and_df
from mlprojects.production.tf_trimodel_extended import tf_trimodel_extended
from api.utils.metrics import accuracy_from_df
import duckdb
from datetime import timedelta, datetime
from airflow.sensors.external_task import ExternalTaskSensor

# Define default arguments for the DAG
default_args = {
    "owner": "mar24cmlops",
    "retries": 2,
    "retry_delay": timedelta(minutes=10),
}


def parse_model_list():
    """
    Parses the model_list.txt file to retrieve production and staging model names.

    :return: A dictionary with keys "production" and "staging" mapping to respective model names.
    """
    models = {}
    with open(resolve_path("models/model_list.txt"), "r") as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            model_type, model_name, version = line.split(",")
            models[model_type.strip()] = f"{model_name.strip()}_{version.strip()}"
    return models


def evaluate(**kwargs):
    """
    Evaluates the performance of the production and staging models using the last 5 days of rolling metrics.

    :param kwargs: context arguments provided by Airflow
    """
    load_dotenv(".envp/.env.airflow")

    # Define the path to the DuckDB database
    duckdb_path = os.path.join(
        resolve_path(os.environ["DATA_PATH"]),
        os.environ["RAKUTEN_DB_NAME"].lstrip("/"),
    )

    # Connect to the DuckDB database
    conn = duckdb.connect(database=duckdb_path, read_only=True)

    # Retrieve the last 5 days of rolling metrics
    rolling_metrics_query = """
    SELECT date, model_name, rolling_mean, rolling_std, rolling_var
    FROM rolling_metrics
    ORDER BY date DESC
    LIMIT 10
    """
    rolling_metrics_df = conn.sql(rolling_metrics_query).df()

    if rolling_metrics_df.empty:
        print("No data found in rolling_metrics table.")
        return

    # Ensure initialization of count_staging_better
    count_staging_better = 0

    # Parse model names
    models = parse_model_list()
    prod_model_name = models.get("production")
    staging_model_name = models.get("staging")

    # Filter data for the last 5 days for each model
    prod_metrics = rolling_metrics_df[
        rolling_metrics_df["model_name"] == prod_model_name
    ].head(5)
    staging_metrics = rolling_metrics_df[
        rolling_metrics_df["model_name"] == staging_model_name
    ].head(5)

    # Calculate the mean rolling mean for the last 5 days
    prod_mean_accuracy = prod_metrics["rolling_mean"].mean()
    staging_mean_accuracy = staging_metrics["rolling_mean"].mean()

    # Calculate the standard deviation of the rolling mean for the last 5 days
    prod_std_accuracy = prod_metrics["rolling_std"].mean()
    staging_std_accuracy = staging_metrics["rolling_std"].mean()

    def compare_model(
        prod_mean_accuracy,
        prod_std_accuracy,
        staging_mean_accuracy,
        staging_std_accuracy,
        unstable_threshold=0.05,
    ):
        if staging_mean_accuracy > prod_mean_accuracy + unstable_threshold:
            better_model = "staging"
        elif prod_mean_accuracy > staging_mean_accuracy + unstable_threshold:
            better_model = "production"
        else:
            better_model = (
                "staging" if prod_std_accuracy > staging_std_accuracy else "production"
            )
        return better_model

    better_model = compare_model(
        prod_mean_accuracy,
        prod_std_accuracy,
        staging_mean_accuracy,
        staging_std_accuracy,
    )

    # Count how many times the staging model outperforms the production model
    count_staging_better = sum(
        staging_metrics["rolling_mean"] > prod_metrics["rolling_mean"]
    )

    if prod_mean_accuracy >= 0.5:
        decision = "production"
        print("Production model has satisfactory performance, no change needed")
    elif prod_mean_accuracy < 0.5 and better_model == "production":
        decision = "retrain"
        print(
            "Production and staging models have unsatisfactory performances, retrain the production model"
        )
    else:
        decision = "staging"
        print(
            f"Production model has unsatisfactory performance and staging performs better, consider staging to production : Staging outperforms production model {count_staging_better} times out of 5 days"
        )

    # Push values to XCom
    kwargs["ti"].xcom_push(key="decision", value=decision)
    kwargs["ti"].xcom_push(key="count_staging_better", value=count_staging_better)


def retrain(**kwargs):
    """
    Retrains the model based on the decision made in the evaluate function.

    :param kwargs: context arguments provided by Airflow
    """
    # Pull values from XCom
    decision = kwargs["ti"].xcom_pull(key="decision", task_ids="evaluate")
    count_staging_better = kwargs["ti"].xcom_pull(
        key="count_staging_better", task_ids="evaluate"
    )

    # Notify if staging model is consistently better
    if decision == "production":
        print("Production model has satisfactory performance, no change needed")
    elif decision == "staging":
        print(
            f"Production model has unsatisfactory performance and staging performs better, consider staging to production : Staging outperforms production model {count_staging_better} times out of 5 days"
        )
        # script pour renvoyer un mail de suggestion
    else:
        print(
            "Production and staging models have unsatisfactory performances, retraining the production model..."
        )
        # script pour renvoyer un mail d'information
        # script de réentrainement
        # script pour envoyer un mail concernant le réentrainement


# Define the DAG with arguments and schedule interval
with DAG(
    dag_id="model_evaluate_and_retrain_dag",
    default_args=default_args,
    description="Evaluate and retrain models based on performance metrics",
    start_date=days_ago(2),
    schedule_interval="*/10 * * * *",  # Run every 10 minutes
    catchup=False,
    tags=["marc24mlops"],
) as dag:
    wait_for_sim_dag = ExternalTaskSensor(
        task_id="wait_for_sim_dag",
        external_dag_id="sim_dag",
        external_task_id="end_task",
        mode="poke",
        timeout=600,
        poke_interval=60,
        retries=5,
        retry_delay=timedelta(minutes=1),
    )

    # Define task to evaluate model performance
    evaluate = PythonOperator(
        task_id="evaluate",
        python_callable=evaluate,
        provide_context=True,
    )

    # Define task to retrain the model if needed
    retrain = PythonOperator(
        task_id="retrain",
        python_callable=retrain,
        provide_context=True,
    )

# Set the task dependencies
wait_for_sim_dag >> evaluate >> retrain
