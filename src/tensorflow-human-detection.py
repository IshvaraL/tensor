# Code adapted from Tensorflow Object Detection Framework
# https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb
# Tensorflow Object Detection Detector
import cv2
import numpy as np
import multiprocessing as mp
import threading as th
import time
from Stream import Stream
from Communication import Comm
from Calibration import Calibration
from DetectorAPI import DetectorAPI
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

CropOffset = 60
threshold = 0.8 # threshold voor herkennen vanmensen.   0 is alles, 1 is niks

class Main:

    def __init__(self):
        self.stream_parent_conn, self.stream_child_conn = mp.Pipe()

        model_path = '../data/faster_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb'
        self.odapi = DetectorAPI(path_to_ckpt=model_path)

        self.cal = Calibration()
        self.comm = Comm()

        self.stream = Stream(self.stream_child_conn)
        self.str = mp.Process(target=self.stream.run, args=())
        self.eat = None
        self.socket_update  = None

        self.frame_lock = mp.RLock()

        self.coords_lock = mp.RLock()

        self.coords = []


    def totuple(self, a):
        try:
            return tuple(self.totuple(i) for i in a)
        except TypeError:
            return a

    def start(self):
        self.str.start()
        self.eat = th.Thread(target=self.eat_rest, name="eat")
        self.socket_update = th.Thread(target=self.update_game, name="update")
        self.eat.start()
        self.socket_update.start()

        ref = cv2.imread('../pics/soccerfield_2d.png')
        height, width, cols = ref.shape
        self.h = None
        pts_src = None
        refClean = ref.copy()

        self.frame_lock.acquire()
        img = self.stream_parent_conn.recv()
        self.frame_lock.release()

        try:
            pts_src = np.load("../data/calibrated_3D.npy")
            pts_dst = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
            self.h, status = cv2.findHomography(pts_src, pts_dst)
        except Exception as e:
            print(e)
            self.coords_3d = self.cal.manual_calibrate(img)
            if len(self.coords_3d) is 4:
                pts_dst = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
                pts_src = np.float32(self.coords_3d)
                self.h, status = cv2.findHomography(pts_src, pts_dst)
                np.save("../data/calibrated_3D", pts_src)

        coords = []

        while True:
            if not self.str.is_alive():
                print("str died")
                break

            start_time = time.time()
            self.frame_lock.acquire()
            img = self.stream_parent_conn.recv()
            self.frame_lock.release()

            x, y, w, h = cv2.boundingRect(pts_src)
            img = img[y-CropOffset:y + h, x:x + w].copy()

            boxes, scores, classes, num = self.odapi.processFrame(img)
            # Visualization of the results of a detection.

            ref = refClean.copy()
            for i in range(len(boxes)):
                # Class 1 represents human
                if classes[i] == 1 and scores[i] > threshold:
                    if len(coords) > 8:
                        break
                    box = boxes[i]
                    cv2.rectangle(img, (box[1], box[0]), (box[3], box[2]), (255, 0, 0), 2)
                    w = box[3] - box[1]
                    h = box[2] - box[0]
                    middle_point = int(box[1]+(w/2)), int(box[2]-h/2)
                    cv2.circle(img, (middle_point), 4, (0, 0, 255), 2)
                    a = np.array([[middle_point[0]+x, middle_point[1]+y-CropOffset+25]], dtype='float32')  # 25 pixels naar beneden om te compenseren voor de halve lengte persoon
                    a = np.array([a])

                    relative_coords = cv2.perspectiveTransform(a, self.h)

                    rx, ry = self.totuple(relative_coords[0][0])
                    cv2.circle(ref, (rx, ry), 10, (0, 0, 255), 4)
                    padding = 1.15
                    if rx > (width-(width*padding)) and rx < (width*padding) and ry > (height-(height*padding)) and ry < (height*padding):
                        coords.append(((rx/width) * 16, 9-((ry/height)*9)))

            #cv2.imshow("transform", ref)
            cv2.imshow("preview", img)

            for x in range(len(coords), 8, 1):
                coords.append((0,0))

            self.coords_lock.acquire()
            self.coords = coords
            self.coords_lock.release()

            coords = []

            end_time = time.time()
            #print("Elapsed Time:", end_time - start_time)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.str.terminate()
                self.str.join()
                break

        self.eat.join()
        self.socket_update.join()
        return

    def eat_rest(self):
        print("Eating started...")
        while True:
            self.frame_lock.acquire()
            x = self.stream_parent_conn.recv()
            self.frame_lock.release()

            if not self.str.is_alive():
                break

    def update_game(self):
        print("Update over socket started...")
        com = Comm()
        coords = []

        while True:
            if com.open():
                while True:
                    self.coords_lock.acquire()
                    coords = self.coords
                    self.coords_lock.release()

                    if len(coords) > 8:
                        coords = coords[0:8]

                    elif len(coords) < 8:
                        for x in range(len(coords), 8, 1):
                            coords.append((0, 0))

                    com.send(coords)
                    time.sleep(0.1)
                    if not self.str.is_alive():
                        break
            com.close()
            if not self.str.is_alive():
                break
        com.close()


if __name__ == "__main__":
    main = Main()
    main.start()
