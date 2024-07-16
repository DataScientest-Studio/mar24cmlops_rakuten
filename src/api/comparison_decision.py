#from api.utils.resolve_path import resolve_path
import numpy as np

def load():
    results=None
    #with open(resolve_path('src/estimation.txt'),'r') as file :
    with open('estimation.txt','r') as file :  # Attention au nom et à l'adresse
        results=file.readlines()
    results=np.array([float(x.split(':')[-1].strip()) for x in results])
    return results  

def spurious(results):
    if np.abs(np.mean(results[-5:])-np.mean(results[:-5])) <0.05:
        return False
    else:
        return True
    
def slope(results):
        y_results=results
        x_ref=np.arange(0,len(results),1)
        return np.cov(x_ref,y_results)[1][0]/np.var(x_ref)

def st_deviation(results):
    return np.std(np.array(results))

def execute():
    results=load()
    if len(results)>15: # sans intérêt si pas assez de données
        if st_deviation(results)>0.05:
            return True
        #if np.abs((slope(results[-5:]-slope(results[:-5]))/slope(results[:-5])))>0.05 : # 5% d'écart
        if slope(results[-5:])<-0.05:
            return True
        if spurious(results):
            return True
    return False

if __name__=='__main__':
    execute()