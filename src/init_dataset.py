# # from api.utils.make_db import init_db
# from api.utils.resolve_path import resolve_path
# from dotenv import load_dotenv
# from api.utils.get_models import upload_model_to_repository
# import os

# env_path = resolve_path(".envp/.env.development")
# load_dotenv(env_path)
# aws_config_path = resolve_path(os.environ["AWS_CONFIG_PATH"])

# # init_db(duckdb_path= resolve_path('data/rakuten_db.duckdb'))

# upload_model_to_repository(
#     aws_config_path,
#     "/home/jc/mar24cmlops_rakuten/models/production_model/tf_trimodel/20240708_19-15-54",
#     "tf_trimodel",
#     is_production=True,
#     version="20240708_19-15-54",
# )
# upload_model_to_repository(
#     aws_config_path,
#     "/home/jc/mar24cmlops_rakuten/models/staging_models/tf_trimodel/20240708_18-02-05",
#     "tf_trimodel",
#     is_production=False,
#     version="20240708_18-02-05",
# )
