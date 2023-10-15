import cv2 as cv
import numpy as np

# Provide the HSV range of color to be detected as marker. 
h_min = 62
h_max = 90
s_min = 22
s_max = 164
v_min = 55
v_max = 255

video_capture = cv.VideoCapture(0)
marker_pts = []

output_file = "marker_output.mp4"
fps = video_capture.get(cv.CAP_PROP_FPS)
f_width = int(video_capture.get(cv.CAP_PROP_FRAME_WIDTH))
f_height = int(video_capture.get(cv.CAP_PROP_FRAME_HEIGHT))
fourcc = cv.VideoWriter_fourcc(*"mp4v")
video_rec = cv.VideoWriter(output_file, fourcc, fps, (f_width, f_height), True)


def getMask(frame, h_min, s_min, v_min, h_max, s_max, v_max):
    frame_hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])
    mask = cv.inRange(frame_hsv, lower, upper)
    return mask


def getContour(mask, marker_pts):
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    for contour in contours:
        area = cv.contourArea(contour)
        if area > 200:
            M = cv.moments(contour)
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            marker_pts.append([cx, cy])


def drawMarker(frame, marker_pts, radius):
    for pt in marker_pts:
        cv.circle(frame, pt, radius, (0, 255, 0), cv.FILLED)


while True:
    success, frame = video_capture.read()

    frame = cv.flip(frame, 1)

    if success is False or cv.waitKey(1) == ord("q"):
        break

    mask = getMask(frame, h_min, s_min, v_min, h_max, s_max, v_max)
    getContour(mask, marker_pts)

    drawMarker(frame, marker_pts, 10)

    cv.imshow("Bruh", frame)
    video_rec.write(frame)


video_capture.release()
video_rec.release()
