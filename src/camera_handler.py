import cv2

def init_cameras():
    left_cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    right_cam = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    return left_cam, right_cam

def get_frames(left_cam, right_cam):
    left_ret, left_frame = left_cam.read()
    right_ret, right_frame = right_cam.read()

    if not left_ret or not right_ret:
        return None, None

    return cv2.flip(left_frame, 1), cv2.flip(right_frame, 1)
