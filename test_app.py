from fastapi.testclient import TestClient
from main import app

# test to check the correct functioning of the /ping route
def test_ping():
    with TestClient(app) as client:
        response = client.get("/hackathon")
        # asserting the correct response is received
        assert response.status_code == 200
        assert response.json() == {"hackathon": "successful"}


# test to check if it is correctly
def test_cs1():
    # defining a sample payload for the testcase
    payload = {
      "Duration_1": 0,
      "Credit_1": 0,
      "Installment_1": 0,
      "Presentresident_1": 0,
      "Age_1": 0,
      "credit_1": 0,
      "Numberofpeoplebeingliable_1": 0
    }
    with TestClient(app) as client:
        response = client.post("/cs1", json=payload)
        # asserting the correct response is received
        assert response.status_code == 200