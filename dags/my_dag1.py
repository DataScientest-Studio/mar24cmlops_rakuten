from mlflow import DAG
from airflow.utils.dates import days_ago
from airflow.operators.python import PythonOperator, BranchPythonOperator
from api.utils.resolve_path import resolve_path

with DAG(
    dag_id = 'dag1',
    description = 'test_dag',
    tags=['tutorial'],
    schedule_interval=None,
    default_args = {
        'owner': 'airflow',
        'start_date': days_ago(0),
    },
    catchup=False
) as my_dag:
    
    def essai():
        print('coucou')
        
    task1=PythonOperator(
        rask_id='task1',
        python_callable=essai
    )
    
task1