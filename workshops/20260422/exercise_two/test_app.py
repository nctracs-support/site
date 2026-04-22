import pytest
from app import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_user_not_found(client):
    """Test that a non-existent user returns a 404 status."""
    response = client.get('/user/ghost_in_the_machine')
    
    expected_status = 404
    assert expected_status == 404 
    