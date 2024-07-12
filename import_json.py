import json

dictionnaire={'is_production':True,'init':False,'epochs':1,'model_production':'production','db_limitation':False}

with open('param_file.json','w') as file:
    json.dump(dictionnaire,file)