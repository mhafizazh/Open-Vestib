import cv2
import numpy as np

def detect_pupil(roi_frame, original_width, original_height):
    gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, threshold = cv2.threshold(blurred, 35, 255, cv2.THRESH_BINARY_INV)
    threshold = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None, None, threshold

    largest = max(contours, key=cv2.contourArea)
    M = cv2.moments(largest)
    if M["m00"] == 0:
        return None, None, threshold

    cX = int(M["m10"] / M["m00"]) + (original_width - roi_frame.shape[1]) // 2
    cY = int(M["m01"] / M["m00"]) + (original_height - roi_frame.shape[0]) // 2

    return cX, cY, threshold
