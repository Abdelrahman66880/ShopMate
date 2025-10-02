import os
from dotenv import load_dotenv

app = Flask(__name__)

@app.route('/')
def home():
    return "Hello World! Welcome to ShopMate."

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)