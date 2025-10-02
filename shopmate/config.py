# config file
import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv(r"/home/stranger/Desktop/ShopMate/.env")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

#paths
DATA_PRODUCT_PATH = r"/home/stranger/Desktop/ShopMate/data/products.db"
DATA_TEXT_PATH = r"/home/stranger/Desktop/ShopMate/data/policies.txt"
STORE_DIRECTORY = r"/home/stranger/Desktop/ShopMate/data/processed"

# Embeddings
load_dotenv(r"/home/stranger/Desktop/ShopMate/.env")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
EMBEDDINGS = GoogleGenerativeAIEmbeddings(model="models/embedding-001")