import cv2
import pandas as pd
import numpy as np


image = cv2.imread(
    r"C:\Users\zuraj\OneDrive\Desktop\OpenCV\Document Pics\11.jpg", cv2.IMREAD_COLOR
)

color_dataset = pd.read_csv(
    r"C:\Users\zuraj\OneDrive\Desktop\OpenCV\Datasets\color_names.csv"
)

image = cv2.resize(image, (500, 1000))

w1 = cv2.namedWindow("Window1")
w2 = cv2.namedWindow("Window2")

cv2.imshow("Window1", image)
image_bg = np.ones(shape=(500, 800, 3), dtype=np.uint8)
image_bg = cv2.putText(
    img=image_bg,
    text="Double Click on Window 1 image",
    org=(55, 250),
    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
    fontScale=1.3,
    color=(255, 255, 255),
    thickness=3,
    lineType=cv2.LINE_AA,
)
cv2.imshow("Window2", image_bg)


def color_finder(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        b, g, r = np.intp(image[y, x, :])
        image_bg = np.ones(shape=(500, 800, 3), dtype=np.uint8)

        key = 10000
        for c_, r_, g_, b_ in color_dataset[
            ["Name", "Red (8 bit)", "Green (8 bit)", "Blue (8 bit)"]
        ].itertuples(index=False):
            d = abs(r_ - r) + abs(g_ - g) + abs(b_ - b)
            if d < key:
                key = d
                c_name = c_

        image_bg = image_bg * np.array([b, g, r], dtype=np.uint8)
        if r + g + b >= 384:
            image_bg = cv2.putText(
                img=image_bg,
                text=c_name,
                org=(50, 200),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=2,
                color=(0, 0, 0),
                thickness=3,
                lineType=cv2.LINE_AA,
            )
        else:
            image_bg = cv2.putText(
                img=image_bg,
                text=c_name,
                org=(50, 200),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=2,
                color=(255, 255, 255),
                thickness=3,
                lineType=cv2.LINE_AA,
            )
        cv2.imshow("Window2", image_bg)


cv2.setMouseCallback("Window1", color_finder)
cv2.waitKey(0)

cv2.destroyAllWindows()
