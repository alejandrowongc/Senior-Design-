#!/usr/bin/env python3
import cv2
import numpy as np
from picamera2 import Picamera2
import time
import RPi.GPIO as GPIO
from RpiMotorLib import RpiMotorLib

# ============================================================
# -------------------- MOTOR SETUP ---------------------------
# ============================================================

DIR_PIN = 22
STEP_PIN = 23
EN_PIN = 24

GPIO.setmode(GPIO.BCM)
GPIO.setup(EN_PIN, GPIO.OUT)

mymotor = RpiMotorLib.A4988Nema(DIR_PIN, STEP_PIN, (21,21,21), "DRV8825")

# Motor state
MOTOR_ACTIVE = False   # False = neutral, True = reject position

STEPS_TO_ACTIVE = 100
STEP_DELAY = 0.0005

def enable_motor():
    GPIO.output(EN_PIN, GPIO.LOW)  # enable driver

def disable_motor():
    GPIO.output(EN_PIN, GPIO.HIGH)  # disable driver (no heat, no holding current)

def motor_to_active():
    global MOTOR_ACTIVE
    if MOTOR_ACTIVE:
        return
    enable_motor()
    mymotor.motor_go(False, "Full", STEPS_TO_ACTIVE, STEP_DELAY, False, 0.05)
    disable_motor()
    MOTOR_ACTIVE = True

def motor_to_neutral():
    global MOTOR_ACTIVE
    if not MOTOR_ACTIVE:
        return
    enable_motor()
    mymotor.motor_go(True, "Full", STEPS_TO_ACTIVE, STEP_DELAY, False, 0.05)
    disable_motor()
    MOTOR_ACTIVE = False


# ============================================================
# ---------------- CAMERA & CALIBRATION ----------------------
# ============================================================

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

# ============================================================
# -------------------- CAMERA SETUP ---------------------------
# ============================================================

picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (view_w, view_h)})
picam2.configure(config)
picam2.start()
time.sleep(2)

picam2.set_controls({"AwbEnable": False, "AeEnable": False})
picam2.set_controls({"AwbMode": 0, "ColourGains": (1.8, 1.0)})

# ============================================================
# ---------------------- ROI SETTINGS -------------------------
# ============================================================

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

# ============================================================
# ----------------------- MAIN LOOP ---------------------------
# ============================================================

while True:
    frame = picam2.capture_array()
    frame = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Draw full ROI on main frame
    cv2.rectangle(frame_rgb, (ROI_X1, ROI_Y1), (ROI_X2, ROI_Y2), (255, 0, 0), 2)

    # Extract ROI
    roi = frame_rgb[ROI_Y1:ROI_Y2, ROI_X1:ROI_X2]

    # Image processing
    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8))

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    decision = None   # PASS or REJECT

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)

        length_pixels = max(w, h)
        length_inches = length_pixels / PIXELS_PER_INCH

        cx, cy = x + w//2, y + h//2

        # Draw measurement box + text
        cv2.rectangle(roi, (x,y), (x+w,y+h), (255,255,0), 1)
        cv2.putText(roi, f"{length_inches:.3f} in | {length_pixels}px",
                    (x, max(0,y-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        cv2.circle(roi, (cx, cy), 3, (255,255,0), -1)

        # Apply pass/reject ONLY if centered
        if center_band_left <= cx <= center_band_right:
            if LOW_LIMIT <= length_inches <= HIGH_LIMIT:
                decision = "PASS"
                cv2.putText(roi, "PASS", (x, y+h+20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            else:
                decision = "REJECT"
                cv2.putText(roi, "REJECT", (x, y+h+20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

    # Draw guidance lines
    cv2.line(roi, (center_band_left, 0), (center_band_left, roi.shape[0]), (0,255,255), 1)
    cv2.line(roi, (center_band_right, 0), (center_band_right, roi.shape[0]), (0,255,255), 1)
    cv2.line(roi, (roi.shape[1]//2,0), (roi.shape[1]//2,roi.shape[0]), (0,255,0), 1)

    # ========================================================
    # ------------------ MOTOR DECISION ----------------------
    # ========================================================

    if decision == "REJECT":
        motor_to_active()

    elif decision == "PASS":
        motor_to_neutral()

    # Display the frame
    cv2.imshow("Full Frame (ROI Marked)", frame_rgb)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ============================================================
# ---------------------- CLEANUP ------------------------------
# ============================================================

disable_motor()
GPIO.cleanup()
cv2.destroyAllWindows()
picam2.stop()
