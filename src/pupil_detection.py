import cv2
import numpy as np

def detect_pupil(roi_frame, original_width, original_height, *, return_size=False):
    """
    If return_size=False (default): returns (x, y, ellipse, threshold)
    If return_size=True:           returns (x, y, size,   threshold)  # like your previous code
    """
    gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, threshold = cv2.threshold(blurred, 35, 255, cv2.THRESH_BINARY_INV)
    threshold = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        # mirror previous behavior: (-1,-1,-1.0) when size requested
        return (-1, -1, -1.0, threshold) if return_size else (None, None, None, threshold)

    largest = max(contours, key=cv2.contourArea)

    # ROI → full-frame offsets (ROI is centered in the full frame)
    ox = (original_width  - roi_frame.shape[1]) // 2
    oy = (original_height - roi_frame.shape[0]) // 2

    ellipse = None
    size = None

    if len(largest) >= 5:
        ellipse = cv2.fitEllipse(largest)  # ((cx,cy), (MA,ma), angle)
        (ecx, ecy), (MA, ma), _ = ellipse
        size = (MA + ma) / 2.0  # mean diameter in pixels
        cX = int(ecx + ox)      # convert ROI center → full-frame
        cY = int(ecy + oy)
    else:
        M = cv2.moments(largest)
        if M["m00"] == 0:
            return (-1, -1, -1.0, threshold) if return_size else (None, None, ellipse, threshold)
        cX = int(M["m10"] / M["m00"]) + ox
        cY = int(M["m01"] / M["m00"]) + oy
        size = -1.0 if return_size else None

    return cX, cY, ellipse, threshold      # original (x, y, ellipse)
