from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import time
import urllib.parse

class PingHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/ping' or self.path == '/ping/':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            response = {
                'status': 'ok',
                'timestamp': time.time(),
                'message': 'Service is running'
            }
            
            self.wfile.write(json.dumps(response).encode('utf-8'))
        else:
            self.send_response(404)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            response = {
                'status': 'error',
                'message': 'Not found'
            }
            
            self.wfile.write(json.dumps(response).encode('utf-8'))

def run_server(port=8080):
    server_address = ('', port)
    httpd = HTTPServer(server_address, PingHandler)
    print(f'Starting ping server on port {port}')
    print('Access http://localhost:8080/ping to test')
    httpd.serve_forever()

if __name__ == '__main__':
    run_server()