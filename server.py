import json
import email
from http.server import BaseHTTPRequestHandler, HTTPServer
from pymongo import MongoClient
from datetime import datetime

import predict

try:
    client = MongoClient("mongodb://localhost:27017/")
    db = client["smart_crop_db"]
    collection = db["predictions"]
    print("Connected to MongoDB successfully!")
except Exception as e:
    print(f"Could not connect to MongoDB: {e}")
    collection = None

class SimpleImageServer(BaseHTTPRequestHandler):
    
    def _send_cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def do_OPTIONS(self):
        self.send_response(200)
        self._send_cors_headers()
        self.end_headers()

    def do_GET(self):
        # Serve the index.html page
        if self.path == '/' or self.path == '/index.html':
            try:
                with open('index.html', 'rb') as file:
                    content = file.read()
                self.send_response(200)
                self.send_header('Content-type', 'text/html')
                self.end_headers()
                self.wfile.write(content)
            except FileNotFoundError:
                self.send_response(404)
                self.end_headers()
                self.wfile.write(b"404 - index.html not found.")
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == '/upload':
            try:
                content_length = int(self.headers.get('Content-Length', 0))
                body = self.rfile.read(content_length)

                msg = email.message_from_bytes(
                    b"Content-Type: " + self.headers['Content-Type'].encode() + b"\r\n\r\n" + body
                )

                file_filename = None
                image_bytes = None

                for part in msg.walk():
                    if part.get_content_maintype() == 'multipart':
                        continue
                    if part.get_param('name', header='content-disposition') == 'file':
                        file_filename = part.get_filename() or 'unknown_image.jpg'
                        image_bytes = part.get_payload(decode=True)
                        break

                if not image_bytes:
                    self.send_response(400)
                    self.end_headers()
                    self.wfile.write(b'{"error": "No file uploaded"}')
                    return

                # Perform prediction using imported logic
                try:
                    disease_name, confidence = predict.predict_disease(image_bytes)
                except Exception as e:
                    self.send_response(500)
                    self.send_header('Content-Type', 'application/json')
                    self._send_cors_headers()
                    self.end_headers()
                    self.wfile.write(json.dumps({"error": str(e)}).encode('utf-8'))
                    return

                # Record outcome to MongoDB
                record = {
                    "filename": file_filename,
                    "predicted_disease": disease_name,
                    "confidence_score": round(confidence, 2),
                    "upload_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                
                if collection is not None:
                    collection.insert_one(record)
                    # Convert ObjectId to string so it can be JSON serialized
                    record['_id'] = str(record['_id'])

                # Return a JSON Response with the result
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self._send_cors_headers()
                self.end_headers()
                self.wfile.write(json.dumps(record).encode('utf-8'))

            except Exception as e:
                self.send_response(500)
                self.end_headers()
                self.wfile.write(f'Internal Server Error: {e}'.encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()

def run(server_class=HTTPServer, handler_class=SimpleImageServer, port=8080):
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    print(f'Starting Pure Python Server on http://localhost:{port} ...')
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    httpd.server_close()
    print('\nServer stopped.')

if __name__ == '__main__':
    run()
