from retrain3mlflow import production_model_retrain
import argparse

parser=argparse.ArgumentParser()
parser.add_argument("--epochs",type=int,default=1)
parser.add_argument("--is_production",type=bool,default=True)
parser.add_argument("--init",type=bool,default=False)
parser.add_argument("--db_limitation",type=bool,default=False)
args=parser.parse_args()
obj=production_model_retrain(model_type="production",is_production=args.is_production)
obj.data_handle(init=args.init,db_limitation=args.db_limitation)
obj.train(epochs=args.epochs)