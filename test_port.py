import socket
import sys

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

if __name__ == "__main__":
    # Test specific ports
    test_ports = [8888, 9999, 3000]
    for port in test_ports:
        if is_port_available(port):
            print(f"Port {port} is AVAILABLE")
        else:
            print(f"Port {port} is IN USE")
    
    # Find an available port in a range
    available_port = find_available_port(8000, 9000)
    if available_port:
        print(f"Found available port: {available_port}")
    else:
        print("No available ports found in range 8000-9000")
        
    # Try to bind to an available port
    try:
        test_port = available_port if available_port else 8080
        print(f"Attempting to bind to port {test_port}...")
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('0.0.0.0', test_port))
            s.listen(1)
            print(f"Successfully bound to port {test_port}")
            print("Press Ctrl+C to exit")
            try:
                # Keep the socket open for a short time
                s.settimeout(10)
                conn, addr = s.accept()
                print(f"Connection from {addr}")
            except socket.timeout:
                print("Timeout waiting for connection")
    except Exception as e:
        print(f"Error binding to port {test_port}: {str(e)}")
        sys.exit(1) 