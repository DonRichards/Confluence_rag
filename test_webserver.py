import socket
import sys
import http.server
import socketserver
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

def is_port_available(port):
    """Check if a port is available for use."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1)
            result = s.connect_ex(('localhost', port))
            return result != 0  # If result is 0, port is in use
    except Exception as e:
        print(f"Error checking port {port}: {str(e)}")
        return False  # Assume not available if there's an error

def find_available_port(start_port, end_port):
    """Find an available port in the given range."""
    for port in range(start_port, end_port + 1):
        if is_port_available(port):
            return port
    return None

class CustomHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(b'<html><body><h1>Test Server Running!</h1></body></html>')

if __name__ == "__main__":
    # Get port from .env file or use default
    try:
        port = int(os.getenv('WEBSITE_PORT', 8888))
        print(f"Using port from .env: {port}")
    except ValueError:
        port = 8888
        print(f"Invalid port in .env, using default: {port}")
    
    # Check if port is available
    if not is_port_available(port):
        print(f"Port {port} is in use, searching for available port...")
        port = find_available_port(8000, 9000)
        if not port:
            print("No available ports found in range 8000-9000")
            sys.exit(1)
        print(f"Found available port: {port}")
    
    # Create HTTP server
    try:
        print(f"Starting HTTP server on port {port}...")
        handler = CustomHTTPRequestHandler
        httpd = socketserver.TCPServer(("", port), handler)
        print(f"Server running at http://localhost:{port}/")
        print("Press Ctrl+C to stop the server")
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        httpd.server_close()
    except Exception as e:
        print(f"Error starting server: {str(e)}")
        sys.exit(1) 