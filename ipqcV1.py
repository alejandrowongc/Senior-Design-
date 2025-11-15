#!/usr/bin/env python3
import cv2
import numpy as np
from picamera2 import Picamera2
import time
import RPi.GPIO as GPIO
from RpiMotorLib import RpiMotorLib

##########################################
# -------- Motor Setup --------
##########################################

DIR_PIN = 22
STEP_PIN = 23
EN_PIN = 24
mymotor = RpiMotorLib.A4988Nema(DIR_PIN, STEP_PIN, (21,21,21), "DRV8825")

GPIO.setmode(GPIO.BCM)
GPIO.setup(EN_PIN, GPIO.OUT)
GPIO.output(EN_PIN, GPIO.LOW)  # enable motor

MOTOR_ACTIVE = False  # motor starts in neutral

STEPS_TO_ACTIVE = 100
STEP_DELAY = 0.0005


def motor_to_active():
    global MOTOR_ACTIVE
    if MOTOR_ACTIVE:
        return
    mymotor.motor_go(False, "Full", STEPS_TO_ACTIVE, STEP_DELAY, False, 0.05)
    MOTOR_ACTIVE = True


def motor_to_neutral():
    global MOTOR_ACTIVE
    if not MOTOR_ACTIVE:
        return
    mymotor.motor_go(True, "Full", STEPS_TO_ACTIVE, STEP_DELAY, False, 0.05)
    MOTOR_ACTIVE = False


##########################################
# -------- Camera Calibration --------
##########################################

cal = np.load("camera_calib.npz")
mtx_full = cal["camera_matrix"]
dist = cal["dist_coeffs"]

orig_w, orig_h = 4056, 3040
view_w, view_h = 800, 480

scale_x = view_w / orig_w
scale_y = view_h / orig_h

mtx_scaled = np.array([
    [mtx_full[0,0] * scale_x, 0,                   mtx_full[0,2] * scale_x],
    [0,                   mtx_full[1,1] * scale_y, mtx_full[1,2] * scale_y],
    [0, 0, 1]
], dtype=np.float32)

new_mtx, _ = cv2.getOptimalNewCameraMatrix(
    mtx_scaled, dist, (view_w, view_h), 1, (view_w, view_h)
)

mapx, mapy = cv2.initUndistortRectifyMap(
    mtx_scaled, dist, None, new_mtx, (view_w, view_h), cv2.CV_32FC1
)

##########################################
# -------- Camera Setup --------
##########################################

picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (view_w, view_h)})
picam2.configure(config)
picam2.start()
time.sleep(2)

picam2.set_controls({"AwbEnable": False, "AeEnable": False})
picam2.set_controls({"AwbMode": 0, "ColourGains": (1.8, 1.0)})

ROI_X1, ROI_Y1, ROI_X2, ROI_Y2 = 10, 175, 790, 325

PIXELS_PER_INCH = 38.5
TARGET_LENGTH = 3.0
TOL = 0.125
LOW_LIMIT = TARGET_LENGTH - TOL
HIGH_LIMIT = TARGET_LENGTH + TOL

roi_w = ROI_X2 - ROI_X1
center_band_left = int((roi_w * 0.5) - (roi_w * 0.10))
center_band_right = int((roi_w * 0.5) + (roi_w * 0.10))

print("Press 'q' to quit")

##########################################
# -------- Main Loop --------
##########################################

while True:
    frame = picam2.capture_array()
    frame = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cv2.rectangle(frame_rgb, (ROI_X1, ROI_Y1), (ROI_X2, ROI_Y2), (255, 0, 0), 2)

    roi = frame_rgb[ROI_Y1:ROI_Y2, ROI_X1:ROI_X2]

    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8))

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    decision = None  # PASS or REJECT or None

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)

        length_pixels = max(w, h)
        length_inches = length_pixels / PIXELS_PER_INCH
        cx = x + w//2

        # Only consider centered parts
        if center_band_left <= cx <= center_band_right:
            if LOW_LIMIT <= length_inches <= HIGH_LIMIT:
                decision = "PASS"
            else:
                decision = "REJECT"

    # Run motor logic based on the decision
    if decision == "REJECT":
        motor_to_active()

    elif decision == "PASS":
        motor_to_neutral()

    cv2.imshow("Frame", frame_rgb)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

GPIO.cleanup()
cv2.destroyAllWindows()
picam2.stop()
