import cv2
import mediapipe as mp
import time
import numpy as np


class SelfieSegmentation:
    def __init__(self, model_selection=0):
        self.modelSelection = model_selection

        self.mpSelfSeg = mp.solutions.selfie_segmentation
        self.mpDraw = mp.solutions.drawing_utils
        self.selfSeg = self.mpSelfSeg.SelfieSegmentation(self.modelSelection)

    def processImage(self, img, bg_img=None, bgBlur=None, bgClr=None, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.selfSeg.process(imgRGB)

        # superimposing selfie segmentation over the background image
        if bgBlur:
            bg_img = cv2.GaussianBlur(img, bgBlur, 0)

        cond = np.stack((self.results.segmentation_mask,) * 3, axis=-1) > 0.1

        # If image not loaded, fill the background with uniform color
        if bgClr:
            bg_img = np.zeros(img.shape, dtype=np.uint8)
            bg_img[:] = bgClr

        if bg_img:
            bg_img = cv2.imread(bg_img)
            bg_img = cv2.resize(bg_img, (1280, 720))

        img = np.where(cond, img, bg_img)

        if draw:
            cv2.imshow("Segmentation Output", img)

        return img


lowBlur = (9, 9)
medBlur = (21, 21)
highBlur = (55, 55)

bgClr1 = (0, 0, 0)
bgClr2 = (255, 255, 255)
bgClr3 = (150, 150, 0)


def main():
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)
    prevTime = 0

    segTool = SelfieSegmentation()

    while True:
        success, img = cap.read()
        img = cv2.flip(img, 1)
        iw, ih, ic = img.shape
        img = segTool.processImage(img, bgBlur=medBlur, draw=False)
        # imPath = "1.png"
        # img = segTool.processImage(img, bg_img=imPath, draw=False)
        # img = segTool.processImage(img, bgClr=bgClr3, draw=False)

        currTime = time.time()
        fps = 1 / (currTime - prevTime)
        prevTime = currTime

        cv2.putText(img, str(f"FPS : {int(fps)}"), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)
        cv2.imshow("Segmentation Output", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
