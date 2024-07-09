from retrain3 import production_model_retrain

obj=production_model_retrain(model_type="production")
obj.data_handle(init=True)
obj.train()