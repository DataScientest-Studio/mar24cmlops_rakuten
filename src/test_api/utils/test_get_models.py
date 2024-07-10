# import pytest
# import os
# from api.utils.get_models import (
#     download_model,
#     verify_and_download_models,
#     list_model_repository_folders,
#     get_model_latest_version,
#     upload_model_to_repository
# )
# from api.utils.resolve_path import resolve_path
# from dotenv import load_dotenv
# from datetime import datetime

# @pytest.fixture(scope="module")
# def setup():
#     """
#     Fixture to setup AWS configuration for tests and create test files.

#     Yields:
#         dict: A dictionary containing paths and configurations for AWS testing.
#     """
#     # Setup
#     local_test_folder = resolve_path("data/tmp")  # Local folder to upload in tests
#     local_download_folder = resolve_path("data/test_dl_model/")  # Local folder for downloaded models

#     # Load environment variables from .env file
#     env_path = resolve_path(".env/.env.development")
#     load_dotenv(env_path)

#     # Convert paths to absolute paths
#     aws_config_path = resolve_path(os.environ["AWS_CONFIG_PATH"])

#     yield {
#         "cfg_path": aws_config_path,
#         "local_test_folder": local_test_folder,
#         "local_download_folder": local_download_folder
#     }

#     # Teardown - Cleanup local test files
#     for folder in [local_test_folder, local_download_folder]:
#         for filename in os.listdir(folder):
#             file_path = os.path.join(folder, filename)
#             if os.path.isfile(file_path):
#                 os.remove(file_path)
#             elif os.path.isdir(file_path):
#                 import shutil
#                 shutil.rmtree(file_path)

# def test_download_model(setup):
#     cfg_path = setup["cfg_path"]
#     local_download_path = setup["local_download_folder"]
#     model_name = "test_model"
#     version = "latest"

#     download_model(cfg_path, model_name, version, local_download_path, is_production=False)

#     # Check if the model files are downloaded
#     assert os.path.exists(local_download_path)
#     assert len(os.listdir(local_download_path)) > 0

# def test_verify_and_download_models(setup):
#     cfg_path = setup["cfg_path"]
#     model_list_file = resolve_path("model_list.txt")

#     with open(model_list_file, "w") as f:
#         f.write("staging,test_model,latest")

#     verify_and_download_models(cfg_path, model_list_file)

#     local_model_path = resolve_path("models/staging_models/test_model/latest")
#     assert len(os.listdir(local_model_path)) > 0

# def test_list_model_repository_folders(setup):
#     cfg_path = setup["cfg_path"]
#     is_production = False

#     model_folders = list_model_repository_folders(cfg_path, is_production)
#     assert "test_model" in model_folders
#     assert len(model_folders["test_model"]) > 0

# def test_get_model_latest_version(setup):
#     cfg_path = setup["cfg_path"]
#     is_production = False
#     model_name = "test_model"

#     latest_version = get_model_latest_version(cfg_path, is_production, model_name)
#     assert latest_version is not None
#     assert isinstance(latest_version, str)

# def test_upload_model_to_repository(setup):
#     cfg_path = setup["cfg_path"]
#     local_model_folder = resolve_path("models/production_model/")
#     model_name = "test_model"
#     is_production = False

#     s3_folder_prefix = upload_model_to_repository(cfg_path, local_model_folder, model_name, is_production)
#     assert s3_folder_prefix.startswith(f"model_repository/staging/{model_name}/")
