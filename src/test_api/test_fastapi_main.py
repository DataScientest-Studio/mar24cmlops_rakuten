from fastapi.testclient import TestClient
from api.fastapi_main import app, lifespan
import pytest
from api.utils.resolve_path import resolve_path

def test_api():
    """
    Test suite for the FastAPI application.
    """
    
    with TestClient(app) as client:
        # Test the root endpoint
        response = client.get("/")
        assert response.status_code == 200
        assert response.json() == {"data": "hello world"}

        # Test the token endpoint with GET method (should be 405 Method Not Allowed)
        response = client.get("/token")
        assert response.status_code == 405

        # Test the token endpoint with POST method for user authentication
        response = client.post("/token", data={"username": "jc", "password": "jc"})
        assert response.status_code == 200
        assert "access_token" in response.json()
        assert response.json()["token_type"] == "bearer"

        # Log in to get the access token
        login_response = client.post("/token", data={"username": "jc", "password": "jc"})
        token = login_response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        # Test the listing_submit endpoint
        with open(resolve_path('data/zazie.jpg'), "rb") as img:
            files = {"image": ("zazie.jpg", img, "image/jpeg")}
            params = {"description": "Livre Zazie dans le métro", "designation": "Livre de Raymond Queneau"}

            response = client.post("/listing_submit", headers=headers, params=params, files=files)

            assert response.status_code == 200
            assert "prediction" in response.json()

        # Extract the listing_id from the response
        listing_id = response.json().get("listing_id")

        # Test the listing_validate endpoint
        params = {"listing_id": listing_id, "user_prdtypecode": "1160"}
        response = client.post("/listing_validate", headers=headers, params=params)
        assert response.status_code == 200
        assert "message" in response.json()

        # Test the read_listing endpoint to verify the listing was updated
        response = client.get(f"/read_listing/{listing_id}", headers=headers)
        assert response.status_code == 200
        response_data = response.json()
        assert response_data.get("user_prdtypecode") is not None
        assert response_data.get("model_prdtypecode") is not None

        # Test the predict_listing endpoint
        params = {"listing_id": 20}
        response = client.post("/predict_listing", headers=headers, params=params)
        assert response.status_code == 200
        assert "prediction" in response.json()

        # Test the delete_listing endpoint
        response = client.delete(f"/delete_listing/{listing_id}", headers=headers)
        assert response.status_code == 200
