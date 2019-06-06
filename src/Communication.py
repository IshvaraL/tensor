import socket
import time
import pickle

HOST = "127.0.0.1"
# PORT = 5005
PORT = 11000


class Comm:

    def __init__(self):
        self.s = None
        self.open()
        self.close()

    def send(self, message):
        data = str(message)
        print(data)
        try:
            self.s.send(data.encode())
            self.s.send(b'<EOF>')
        except Exception as e:
            self.close()
            # time.sleep(0.001)
            self.open()
            print(e)

    def close(self):
        self.s.close()
        self.s = None

    def open(self):
        if self.s is None:
            self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.s.settimeout(1)
            try:
                self.s.connect((HOST, PORT))
                print("Connected")
                return True
            except Exception as e:
                print(e)
                return False
        return True

if __name__ == "__main__":

    com = Comm()
    coords = [(1.1245929509943182, 4.44774710814543), (2.2914888139204543, 1.3934880365182352), (3.620165127840909, 6.051409187963453), (0, 0), (0, 0), (0, 0), (0, 0)]

    if len(coords) > 8:
        coords = coords[0:8]

    elif len(coords) < 8:
        for x in range(len(coords), 8, 1):
            coords.append((0,0))

    if com.open():
        for x in range(0,20,1):
            com.send(coords)
            time.sleep(1)

    com.close()
