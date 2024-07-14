from airflow import DAG
#from airflow.utils.dates import days_ago
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
import datetime

# définir en variable
fictive_current_date=Variable.set(datetime.datetime.strptime("2024-07-01","%Y-%m-%d"))

with DAG(
    dag_id = 'my_very_first_dag',
    description = 'My first DAG created with DataScientest',
    tags=['tutorial', 'datascientest'],
    #schedule_interval='00 00 * * *', # journalier
    schedule_interval = '*/5 * * * * 0', # toutes les 5 minutes (oui,mon ordinateur est lent)
    # Vérifier si problème avec la première estimation (1h)
    default_args = {
        'owner': 'airflow',
        'start_date': datetime.datetime.now(),
    }
) as my_dag:
    
    def calendar():
        
    
    task1 = PythonOperator(
    task_id='calendrier',
    python_callable=calendar,
    #op_kwargs= {
    #    '': ''
    #}
    )