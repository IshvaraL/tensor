import socket
import time
import base64
import random
import pickle

HOST = '127.0.0.1'
PORT = 11000

while True:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen()
        conn, addr = s.accept()
        with conn:
            print('Connected by', addr)
            while True:
                try:
                    data = conn.recv(1024)
                except Exception as e:
                    break
                if not data:
                    break
                print(repr(data))
