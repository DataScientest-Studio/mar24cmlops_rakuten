from airflow import DAG
#from airflow.utils.dates import days_ago
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.bash import BashOperator
from airflow.operators.dummy import DummyOperator
import datetime
from api.utils.resolve_path import resolve_path
import travail2
import comparision_decision

# définir en variable
Variable.set(key='fictive_current_date',value=datetime.datetime.strptime("2024-07-01","%Y-%m-%d"))
Variable.set()
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
        current_date=Variable.get(key='fictive_current_date')
        # mettre un sensor ou autre pour vérifier que l'on peut bien passer à une nouvelle date
        current_date=current_date+datetime.timedelta(days=1)
        Variable.set(key='fictive_current_date',value=current_date)
        
    def production_estimation():
        current_date=Variable.get(key='fictive_current_data')
        #pathway=resolve_path('models/production_model')
        travail2.execution(model_type='production',i_date=current_date)
    
    def decide_branch():
        result=comparison_decision.execute()
        if result is True:
            return 'task4'
        else:
            return 'f_task'
        
    def production_estimation2():
        current_date=Variable.get(key='fictive_current_data')
        #pathway=resolve_path('models/production_model')
        travail2.execution(model_type='staging',i_date=current_date)
        
    task1 = PythonOperator(
        task_id='scheduler',
        python_callable=calendar
    )
    
    task2=PythonOperator(
        task_id = 'Estimation_production',
        python_callable=production_estimation
    )
    
    task3=BranchPythonOperator(
        task_id='branching',
        python_callable=decide_branch,
        #op_args={'condition':bool(random.getrandbits(1))}        
    )
    
    task4=PythonOperator(
        task_id='another model',
        python_callable=production_estimation2        
    )
    
    
    f_task=PythonOperator(
        task_id="Achievement",python_callable
    )
    
task1 >> task2 >> task3
task3 >> [task4,f_task]
task4 >> f_task