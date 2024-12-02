import socket 

host = '127.0.0.1'
port = 8888

while True:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.connect((host, port)) # Connect
        received = sock.recv(4096) # Receive data synchronically
        print('received')