import cv2


class Calibration:

    def __init__(self):
        self.img = 0
        self.cal_coords = []

    def draw_circle3d(self, event, x, y, flags, param):
        global ix, iy
        # global cal_coords
        # global img
        if event == cv2.EVENT_LBUTTONDBLCLK:
            cv2.circle(self.img, (x, y), 3, (255, 0, 0), -1)
            ix, iy = x, y
            self.cal_coords.append((ix, iy))

    def manual_calibrate(self, frame):
        # global cal_coords
        self.cal_coords = []
        # global img
        self.img = frame
        cv2.namedWindow('3d image')
        cv2.setMouseCallback('3d image', self.draw_circle3d)
        cv2.imshow("3d image", self.img)
        while len(self.cal_coords) < 4:
            cv2.imshow("3d image", self.img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        return self.cal_coords
