import pytest
import os
from cryptography.fernet import Fernet
from passlib.hash import bcrypt
from datetime import datetime, timedelta
import jwt
from api.utils.security import generate_key, load_key, encrypt_file, decrypt_file, verify_password, create_access_token

@pytest.fixture(scope='module')
def setup():
    """
    Fixture for setup and teardown actions.
    """
    test_file = 'test.txt'
    encrypted_file = 'test_encrypted.txt'
    decrypted_file = 'test_decrypted.txt'
    secret_key_file = 'fernet_key.txt'
    plain_password = 'securepassword123'
    hashed_password = bcrypt.hash(plain_password)
    jwt_data = {'sub': '1234567890', 'name': 'John Doe', 'admin': True}
    os.environ['JWT_KEY'] = Fernet.generate_key().decode()
    os.environ['ALGORITHM'] = 'HS256'

    # Create a sample file to encrypt
    with open(test_file, 'w') as f:
        f.write('This is a test file.')

    yield {
        'test_file': test_file,
        'encrypted_file': encrypted_file,
        'decrypted_file': decrypted_file,
        'secret_key_file': secret_key_file,
        'plain_password': plain_password,
        'hashed_password': hashed_password,
        'jwt_data': jwt_data
    }

    # Clean up files created during tests
    os.remove(test_file)
    os.remove(encrypted_file)
    os.remove(decrypted_file)
    os.remove(secret_key_file)

def test_generate_key(setup):
    generate_key()
    assert os.path.exists(setup['secret_key_file'])

def test_load_key(setup):
    key = load_key()
    assert isinstance(key, bytes)

def test_encrypt_file(setup):
    key = load_key()
    encrypt_file(key, setup['test_file'], setup['encrypted_file'])
    assert os.path.exists(setup['encrypted_file'])
    with open(setup['encrypted_file'], 'rb') as f:
        encrypted_data = f.read()
    assert encrypted_data != b'This is a test file.'

def test_decrypt_file(setup):
    key = load_key()
    decrypt_file(key, setup['encrypted_file'], setup['decrypted_file'])
    assert os.path.exists(setup['decrypted_file'])
    with open(setup['decrypted_file'], 'r') as f:
        decrypted_data = f.read()
    assert decrypted_data == 'This is a test file.'

def test_verify_password(setup):
    assert verify_password(setup['plain_password'], setup['hashed_password'])
    assert not verify_password('wrongpassword', setup['hashed_password'])

def test_create_access_token(setup):
    token = create_access_token(setup['jwt_data'])
    decoded_data = jwt.decode(token, os.environ['JWT_KEY'], algorithms=[os.environ['ALGORITHM']])
    assert decoded_data['sub'] == '1234567890'
    assert decoded_data['name'] == 'John Doe'
    assert decoded_data['admin']
    assert 'exp' in decoded_data