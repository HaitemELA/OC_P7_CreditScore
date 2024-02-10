# test_dash_app.py

import pytest
import json
import gc
from flask import Flask
from api.app import app, predict

@pytest.fixture
def test_app():
    return app

def test_predict(test_app):
    # Define the expected input data for the test
    test_client_id = '100001'
    api_url = "/predict"  # Use relative path
    json_data = json.dumps({"data": {"SK_ID_CURR": test_client_id}})
    
    # Monkeypatch any external dependencies if needed
    # monkeypatch.setattr("your_module.external_dependency", mocked_external_dependency)

    # Use the Flask test client to simulate a request to the predict endpoint
    with test_app.test_client() as client:
        response = client.post(api_url, data=json_data, headers={"Content-Type": "application/json"})
    # Assertions
    assert response.status_code == 200
    assert int(response.json['decision']) == 0
    print(response.json)


def test_client_info(test_app):
    test_client_id = '100001'
    api_url = "/client"

    json_data = json.dumps({"data": {"SK_ID_CURR": test_client_id}})

    with test_app.test_client() as client:
        response = client.post(api_url, data=json_data, headers={"Content-Type": "application/json"})
        print(response)
    # Assertions
    if response.status_code == 200:
        assert response.status_code == 200, "Success: Status code 200"
        assert response.json['Age'] == '52 years old', "Success: Age is 52 years old"
    