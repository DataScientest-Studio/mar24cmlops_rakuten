import logging
from datetime import datetime
from api.utils.resolve_path import resolve_path

# Configurer les journaux
logging.basicConfig(level=logging.INFO, format='%(message)s', handlers=[
    logging.FileHandler(resolve_path("logs/user.log")),
    logging.FileHandler(resolve_path("logs/product.log")),
    logging.StreamHandler()
])

def log_user_action(endpoint_name, username, return_code, access_rights):
    """
    Journalise les actions des utilisateurs.

    Args:
        endpoint_name (str): Le nom de l'endpoint.
        username (str): Le nom d'utilisateur.
        return_code (int): Le code de retour de la réponse.
        access_rights (str): Les droits de l'utilisateur.
    """
    log_entry = f"{datetime.now()}, [{endpoint_name}], {username}, {return_code}, {access_rights}"
    logging.info(log_entry)

def log_product_action(endpoint_name, return_code, user_id, listing_id, model_prdtypecode=None, user_prdtypecode=None):
    """
    Journalise les actions sur les produits.

    Args:
        endpoint_name (str): Le nom de l'endpoint.
        return_code (int): Le code de retour de la réponse.
        user_id (int): L'ID de l'utilisateur.
        listing_id (int): L'ID de la liste.
        model_prdtypecode (int, optional): Le code de produit du modèle. Par défaut à None.
        user_prdtypecode (int, optional): Le code de produit de l'utilisateur. Par défaut à None.
    """
    log_entry = f"{datetime.now()}, [{endpoint_name}], {return_code}, {user_id}, {listing_id}, {model_prdtypecode}, {user_prdtypecode}"
    logging.info(log_entry)
