# from dotenv import load_dotenv
# from api.utils.resolve_path import resolve_path
# import os
# import duckdb

# load_dotenv(".envp/.env.development")

# duckdb_path = os.path.join(
#     resolve_path(os.environ["DATA_PATH"]),
#     os.environ["RAKUTEN_DB_NAME"].lstrip("/"),
# )

# new_conn = duckdb.connect(
#     database=duckdb_path, read_only=True)

# cols = [
#     "designation",
#     "description",
#     "productid",
#     "imageid",
#     "user_prdtypecode",
#     "model_prdtypecode",
#     "waiting_datetime",
#     "validate_datetime",
#     "status",
#     "user",
#     "imageid",
# ]
# columns_str = ", ".join(cols)
# result = new_conn.sql(
#     f"SELECT {columns_str} FROM fact_listings WHERE validate_datetime >= DATETIME '2024-07-17' AND validate_datetime <= DATETIME '2024-07-18'"
# ).df()

# print(result)
