from cryptography.fernet import Fernet
from passlib.hash import bcrypt
from datetime import datetime, timedelta
import jwt
import os


def generate_key():
    """
    Generate a Fernet key and save it to a file.
    """
    key = Fernet.generate_key()
    with open("fernet_key.txt", "wb") as key_file:
        key_file.write(key)


def load_key():
    """
    Load the Fernet key from a file.
    """
    with open("fernet_key.txt", "rb") as key_file:
        key = key_file.read()
    return key


def encrypt_file(key, input_file, output_file):
    """
    Encrypt a file using the provided key.

    Args:
        key (bytes): The encryption key.
        input_file (str): The path to the input file.
        output_file (str): The path to save the encrypted file.
    """
    fernet = Fernet(key)
    with open(input_file, "rb") as file:
        data = file.read()
    encrypted_data = fernet.encrypt(data)
    with open(output_file, "wb") as file:
        file.write(encrypted_data)


def decrypt_file(key, input_file, output_file):
    """
    Decrypt a file using the provided key.

    Args:
        key (bytes): The encryption key.
        input_file (str): The path to the encrypted file.
        output_file (str): The path to save the decrypted file.
    """
    fernet = Fernet(key)
    with open(input_file, "rb") as file:
        encrypted_data = file.read()
    decrypted_data = fernet.decrypt(encrypted_data)
    with open(output_file, "wb") as file:
        file.write(decrypted_data)


def verify_password(plain_password, hashed_password):
    """
    Verifies if the plain text password matches the hashed password.

    Args:
    - plain_password (str): Plain text password to verify.
    - hashed_password (str): Hashed password to compare.

    Returns:
    - bool: True if the password matches, False otherwise.
    """
    return bcrypt.verify(plain_password, hashed_password)


def create_access_token(data: dict, expires_delta: timedelta = None):
    """
    Create a JWT access token.

    Args:
        data (dict): The data to encode in the token.
        expires_delta (timedelta, optional): The token expiry duration. Defaults to 15 minutes if not provided.

    Returns:
        str: The encoded JWT token.
    """
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(
        to_encode, os.environ["JWT_KEY"], algorithm=os.environ["ALGORITHM"]
    )
    return encoded_jwt
