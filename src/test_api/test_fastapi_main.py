from fastapi.testclient import TestClient
from api.fastapi_main import app

client = TestClient(app)


def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"data": "hello world"}

# def test_login_for_access_token():
#     response = client.post("/token", data={"username": "testuser", "password": "testpassword"})
#     assert response.status_code == 200
#     assert "access_token" in response.json()
#     assert response.json()["token_type"] == "bearer"

# def test_read_listing():
#     # Simuler la connexion pour obtenir le token
#     login_response = client.post("/token", data={"username": "testuser", "password": "testpassword"})
#     token = login_response.json()["access_token"]
#     headers = {"Authorization": f"Bearer {token}"}

#     # Test du endpoint read_listing
#     response = client.get("/read_listing/1", headers=headers)
#     assert response.status_code == 200
#     assert "designation" in response.json()

# def test_delete_listing():
#     # Simuler la connexion pour obtenir le token
#     login_response = client.post("/token", data={"username": "testuser", "password": "testpassword"})
#     token = login_response.json()["access_token"]
#     headers = {"Authorization": f"Bearer {token}"}

#     # Test du endpoint delete_listing
#     response = client.delete("/delete_listing/1", headers=headers)
#     assert response.status_code == 200
#     assert response.json() == {"message": "Listing deleted successfully"}

# def test_listing_submit():
#     # Simuler la connexion pour obtenir le token
#     login_response = client.post("/token", data={"username": "testuser", "password": "testpassword"})
#     token = login_response.json()["access_token"]
#     headers = {"Authorization": f"Bearer {token}"}

#     # Test du endpoint listing_submit
#     with open("path/to/test/image.jpg", "rb") as img:
#         files = {"image": img}
#         data = {"description": "test description", "designation": "test designation"}
#         response = client.post("/listing_submit", headers=headers, data=data, files=files)
#         assert response.status_code == 200
#         assert "listing_id" in response.json()

# def test_listing_validate():
#     # Simuler la connexion pour obtenir le token
#     login_response = client.post("/token", data={"username": "testuser", "password": "testpassword"})
#     token = login_response.json()["access_token"]
#     headers = {"Authorization": f"Bearer {token}"}

#     # Test du endpoint listing_validate
#     data = {"listing_id": 1, "user_prdtypecode": 123}
#     response = client.post("/listing_validate", headers=headers, json=data)
#     assert response.status_code == 200
#     assert response.json() == {"message": f"Listing 1 validated successfully with user_prdtypecode 123"}

# def test_predict_listing():
#     # Simuler la connexion pour obtenir le token
#     login_response = client.post("/token", data={"username": "testuser", "password": "testpassword"})
#     token = login_response.json()["access_token"]
#     headers = {"Authorization": f"Bearer {token}"}

#     # Test du endpoint predict_listing
#     data = {"listing_id": 1}
#     response = client.post("/predict_listing", headers=headers, json=data)
#     assert response.status_code == 200
#     assert "prediction" in response.json()
