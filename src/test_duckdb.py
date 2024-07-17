# import os
# from dotenv import load_dotenv
# from api.utils.resolve_path import resolve_path

# # from api.utils.make_db import process_listing
# # from api.utils.predict import load_models_from_file, predict_from_model_and_df
# # from mlprojects.production.tf_trimodel_extended import tf_trimodel_extended
# # from api.utils.metrics import accuracy_from_df
# import duckdb

# # def retrieve_data_and_predict():
# load_dotenv(".envp/.env.development")

# duckdb_path = os.path.join(
#     resolve_path(os.environ["DATA_PATH"]),
#     os.environ["RAKUTEN_DB_NAME"].lstrip("/"),
# )

# conn = duckdb.connect(database=duckdb_path, read_only=True)
# result = conn.sql("SELECT * FROM accuracy_daily limit 10").df()

# print(result)

# result = conn.sql("SELECT * FROM rolling_metrics limit 10").df()

# print(result)


# #     cols = [
# #         "designation",
# #         "description",
# #         "productid",
# #         "imageid",
# #         "user_prdtypecode",
# #         "model_prdtypecode",
# #         "waiting_datetime",
# #         "validate_datetime",
# #         "status",
# #         "user",
# #         "imageid",
# #     ]
# #     columns_str = ", ".join(cols)
# #     result = conn.sql(
# #         f"SELECT {columns_str} FROM fact_listings WHERE validate_datetime >= DATETIME '2024-07-17' AND validate_datetime <= DATETIME '2024-07-18'"
# #     ).df()

# #     result["image_path"] = result.apply(
# #         lambda row: resolve_path(
# #             f"data/images/image_train/image_{row['imageid']}_product_{row['productid']}.jpg"
# #         ),
# #         axis=1,
# #     )

# #     ############
# #     models = load_models_from_file(os.environ["AWS_CONFIG_PATH"], resolve_path('models/model_list.txt'))
# #     acc_from_models = {}

# #     for model in models:
# #         colname = f"{model.model_name}_{model.version}_{model.model_type}"
# #         result = predict_from_model_and_df(model, result)
# #         acc_from_models[colname] = accuracy_from_df(result, colname, "user_prdtypecode")

# #     print(acc_from_models)


# # def insert_accuracy_into_db():
# #     print("accuracy inserted")

# # retrieve_data_and_predict()
