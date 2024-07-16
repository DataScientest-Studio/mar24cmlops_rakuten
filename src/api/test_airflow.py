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
Variable.set(key='initialization',value=0)
with DAG(
    dag_id = 'my_very_first_dag',
    description = 'My first DAG',
    tags=['rakuten', 'datascientest'],
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
        state = Variable.get(key='initiaization')
        if state==1:
            return 'f_task'
        current_date=current_date+datetime.timedelta(days=1)
        Variable.set(key='fictive_current_date',value=current_date)
        Variable.set(key='initialization',value=1)
        return 'task1'
        
    def production_estimation(model_type):
        current_date=Variable.get(key='fictive_current_data')
        #pathway=resolve_path('models/production_model')
        # il faut spécifier un nom de fichier (ou le chemin) pour les estimations
        # Penser à avoir un script qui renvoye une valeur !!!
        travail2.execution(model_type='production',i_date=current_date)
    
    def decide_branch():
        result=comparison_decision.execute()
        if result is True:
            return 'task4'
        else:
            return 'f_task'
        
    def retrain():
        # trouver le bon script/module
    
    
    
    def termination():
        Variable.set(key='initialization',value=2)
    
    task1 = BranchPythonOperator(
        task_id='scheduler',
        python_callable=calendar
    )
    
    task2=PythonOperator(
        task_id = 'Estimation_production',
        python_callable=production_estimation,
        op_kwargs= {
        'model_type': 'production'
    }
    )
    
    task3=BranchPythonOperator(
        task_id='branching_1',
        python_callable=decide_branch,    
    )
    
    task4=PythonOperator(
        task_id='another model',
        python_callable=production_estimation,
        op_kwargs={
            'model_type':'staging',      
        }        
    )
    
    task5=BranchPythonOperator(
        task_id='branching_2',
        python_callable=decide_branch,
        # op_kwargs={                  # pour spécifier le chemin du fichier d'estimation
        #     'mmodel_type':'staging'
        # }
        
    )
    
    task6=PythonOperator(
        task_id='retrain',
        python_callable = retrain
    )

    task7=PythonOperator(
        task_id='retrain_estimation',
        python_callable=production_estimation,
                op_kwargs={
            'model_type':'staging',      
        } 
    )
    
    task8=
    
    f_task=PythonOperator(
        task_id="Achievement",
        python_callable=termination
    )
    
task1 >> [task2,f_task]
task2 >> task3
task3 >> [task4,f_task]
task4 >> f_task