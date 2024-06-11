from cryptography.fernet import Fernet

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