from fastapi.testclient import TestClient
from api.fastapi_main import app
from api.fastapi_main import app, lifespan
import pytest
from api.utils.resolve_path import resolve_path

# Création de la fixture pour la connexion DuckDB et l'application FastAPI
# https://stackoverflow.com/questions/75714883/how-to-test-a-fastapi-endpoint-that-uses-lifespan-function

def test_api():
    
    with TestClient(app) as client:
        
        response = client.get("/")
        assert response.status_code == 200
        assert response.json() == {"data": "hello world"}
        
        response = client.get("/token")
        assert response.status_code == 405
        
        response = client.post("/token", data={"username": "jc", "password": "jc"})
        assert response.status_code == 200
        assert "access_token" in response.json()
        assert response.json()["token_type"] == "bearer"
        
        login_response = client.post("/token", data={"username": "jc", "password": "jc"})
        token = login_response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        
        with open(resolve_path('data/zazie.jpg'), "rb") as img:
            files = {"image": ("zazie.jpg", img, "image/jpeg")}
            params = {"description": "Livre Zazie dans le métro", "designation": "Livre de Raymond Queneau"}
            
            response = client.post("/listing_submit", headers=headers, params=params, files=files)
            
            assert response.status_code == 200
            assert "prediction" in response.json()
        
        
        listing_id = response.json().get("listing_id")
        params = {"listing_id": listing_id, "user_prdtypecode": "1160"}
        response = client.post("/listing_validate", headers=headers, params=params)
        assert response.status_code == 200
        assert "message" in response.json()
        
        response = client.get(f"/read_listing/{listing_id}", headers=headers)
        assert response.status_code == 200
        response_data = response.json()
        assert response_data.get("user_prdtypecode") is not None
        assert response_data.get("model_prdtypecode") is not None
        
        params = {"listing_id": 20}
        response = client.post("/predict_listing", headers=headers, params=params)
        assert response.status_code == 200
        assert "prediction" in response.json()
        
        response = client.delete(f"/delete_listing/{listing_id}", headers=headers)
        assert response.status_code == 200