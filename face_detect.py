import cv2
import numpy as np
from helper_functions import Join_Images

THRESHOLD_VALUE = 100

videocap = cv2.VideoCapture(
    0
)

output_file = r"motion_detection\motion_locator_video.mp4"
fps = videocap.get(cv2.CAP_PROP_FPS)
frame_width = int(videocap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(videocap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# fps = 30
# frame_width = 1280
# frame_height = 720


fourcc = cv2.VideoWriter_fourcc(*"mp4v")
video_writer = cv2.VideoWriter(
    output_file, fourcc, fps, (frame_width, frame_height), True
)


def eucledian_dst(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    return np.sqrt(np.square(x2 - x1) + np.square(y2 - y1))


def cluster_closest(full, threshold):
    """
    Input is array of x, y coordinates.
    Goal is to find the eucledian distance between all the points and divide the points into different clusters based on the threshold of distance.
    """

    start = full[:, :2]
    end = full[:, 2:]

    n_points = start.shape[0]

    cluster_start = start.copy().reshape(-1, 1, 2).tolist()
    cluster_end = end.copy().reshape(-1, 1, 2).tolist()

    for i in range(n_points):
        for j in range(n_points):
            pt1 = start[i].tolist()
            pt2 = start[j].tolist()
            pt1_e = end[i].tolist()
            pt2_e = end[j].tolist()
            dst = eucledian_dst(pt1, pt2)
            if dst <= threshold and i != j:
                if len(cluster_start[i]) > 1 and len(cluster_start[j]) > 1:
                    for ele1, ele2 in zip(cluster_start[j], cluster_end[j]):
                        if ele1 not in cluster_start[i]:
                            cluster_start[i].append(ele1)
                            cluster_end[i].append(ele2)
                    cluster_start[j] = []
                    cluster_end[j] = []
                elif len(cluster_start[i]) > 1:
                    if pt2 not in cluster_start[i]:
                        cluster_start[i].append(pt2)
                        cluster_end[i].append(pt2_e)
                    cluster_start[j] = []
                    cluster_end[j] = []
                elif len(cluster_start[j]) > 1:
                    if pt1 not in cluster_start[j]:
                        cluster_start[j].append(pt1)
                        cluster_end[j].append(pt1_e)
                    cluster_start[i] = []
                    cluster_end[i] = []
                else:
                    if len(cluster_start[i]) != 0 and len(cluster_start[j]) != 0:
                        cluster_start[i] = [pt1, pt2]
                        cluster_end[i] = [pt1_e, pt2_e]
                        cluster_start[j] = []
                        cluster_end[j] = []
    return (
        [ele for ele in cluster_start if ele != []],
        [ele for ele in cluster_end if ele != []],
    )


while True:
    success1, frame1 = videocap.read()
    if success1 is not True or cv2.waitKey(5) == ord("q"):
        break
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    success2, frame2 = videocap.read()
    if success2 is not True or cv2.waitKey(5) == ord("q"):
        break
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    success3, frame3 = videocap.read()
    if success3 is not True or cv2.waitKey(5) == ord("q"):
        break
    gray3 = cv2.cvtColor(frame3, cv2.COLOR_BGR2GRAY)

    diff1 = cv2.absdiff(gray3, gray2)
    diff2 = cv2.absdiff(gray2, gray1)
    _, diff1_thresh = cv2.threshold(diff1, THRESHOLD_VALUE, 255, cv2.THRESH_BINARY)
    _, diff2_thresh = cv2.threshold(diff2, THRESHOLD_VALUE, 255, cv2.THRESH_BINARY)

    diff = cv2.dilate(
        cv2.erode(cv2.subtract(diff2_thresh, diff1_thresh), (150, 150), 2),
        (150, 150),
        2,
    )
    cv2.imshow("diff", diff)

    canny = cv2.Canny(diff, threshold1=100, threshold2=100)

    contours, _ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    bound_rect_arr = None
    for contour in contours:
        if cv2.contourArea(contour) > 0:
            x, y, w, h = cv2.boundingRect(contour)
            if bound_rect_arr is None:
                bound_rect_arr = np.array([(x, y, x + w, y + h)])
            bound_rect_arr = np.concatenate(
                [bound_rect_arr, np.array([(x, y, x + w, y + h)])]
            )

    if bound_rect_arr is not None:
        start_cluster, end_cluster = cluster_closest(bound_rect_arr, 400)
        for start, end in zip(start_cluster, end_cluster):
            x_min, y_min = np.array(start).min(axis=0)
            x_max, y_max = np.array(end).max(axis=0)
            cv2.rectangle(frame3, (x_min, y_min), (x_max, y_max), (0, 255, 0), 4)
    cv2.imshow("img", frame3)
    video_writer.write(frame3)

videocap.release()
video_writer.release()
cv2.destroyAllWindows()
