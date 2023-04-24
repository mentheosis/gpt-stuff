import requests
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(dotenv_path="private.env")

def test():
    url = ""
    params = {
        "client_id": app_id,
    }

    response = requests.get(url, params=params)
    print(response.content)
