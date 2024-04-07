# server.py
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import subprocess

import pickle
import re
from pathlib import Path
from transformers import DistilBertTokenizer, DistilBertModel
from torch.utils.data import DataLoader, TensorDataset
import torch

__version__ = "0.1.0"

BASE_DIR = Path(__file__).resolve(strict=True).parent


with open(f"{BASE_DIR}/svm_model-{__version__}.pkl", "rb") as f:
    model = pickle.load(f)

def get_embeddings(texts):
    print("Embedding: " + texts)
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertModel.from_pretrained('distilbert-base-uncased')

    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :]
    
    return embeddings

classes = [
    "Risk",
    "NonRisk"
]


def predict_pipeline(text):
    # text = re.sub(r'[!@#$(),\n"%^*?\:;~`0-9]', " ", text)
    # text = re.sub(r"[[]]", " ", text)
    text = text.lower()
    embeddings = get_embeddings(text)
    pred = model.predict(embeddings)
    print(pred)
    return pred[0]


class RequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed_path = urlparse(self.path)
        query_params = parse_qs(parsed_path.query)
        if 'input' in query_params:
            user_input = query_params['input'][0]
            # Adjust the path to your script.py
            # result = subprocess.check_output(['py', 'D:\SVM_SL\mock\script.py', user_input]).decode().strip()
            output = str(predict_pipeline(user_input))
            print("ABC" + output + "ABC")
            
            # Add CORS headers
            self.send_response(200)
            self.send_header('Content-type', 'text/plain')
            self.send_header('Access-Control-Allow-Origin', '*')  # Allow requests from any origin
            self.end_headers()
            self.wfile.write(output.encode())
        else:
            print("Input not found")
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b'Bad request')

if __name__ == '__main__':
    server_address = ('', 8000)
    httpd = HTTPServer(server_address, RequestHandler)
    print('Starting server on port 8000...')
    httpd.serve_forever()
    # predict_pipeline("Angry emergency fastastic so bad very bad oh my god")
