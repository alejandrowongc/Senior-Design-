#!/usr/bin/env python3
import cv2
import numpy as np
from picamera2 import Picamera2
import time

# ---------------- Camera Setup ----------------
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (800, 600)})
picam2.configure(config)
picam2.start()
time.sleep(2)

# Lock exposure and white balance
controls = {"AwbEnable": False, "AeEnable": False}
picam2.set_controls(controls)
picam2.set_controls({"AwbMode": 0, "ColourGains": (1.8, 1.0)})

# ROI definition
ROI_X1, ROI_Y1, ROI_X2, ROI_Y2 = 10, 175, 790, 325

print("Press 'q' to quit")
while True:
    frame = picam2.capture_array()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Draw ROI on full frame
    cv2.rectangle(frame_rgb, (ROI_X1, ROI_Y1), (ROI_X2, ROI_Y2), (255, 0, 0), 2)

    # Extract ROI
    roi = frame_rgb[ROI_Y1:ROI_Y2, ROI_X1:ROI_X2]

    # Grayscale + blur + edges
    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Process contours
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 1000 or area > 15000:
            # Filter out small noise or large irrelevant blobs
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(roi, f"{w}x{h}px", (x, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        # Optional: centroid for position tracking
        cx, cy = x + w // 2, y + h // 2
        cv2.circle(roi, (cx, cy), 3, (0, 0, 255), -1)

    # Show
    cv2.imshow("Full Frame (ROI Marked)", frame_rgb)
    cv2.imshow("ROI Contours", roi)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
picam2.stop()
