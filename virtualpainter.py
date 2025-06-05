import cv2
import numpy as np
import os
import HandTrackingModule as htm

# -------------------- Configuration --------------------
brushThickness = 15
eraserThickness = 60
canvasSize = (720, 1280)  # height, width

# -------------------- Load Header Images --------------------
folderPath = "Header"
overlayList = [cv2.resize(cv2.imread(os.path.join(folderPath, img)), (1280, 125))
               for img in sorted(os.listdir(folderPath)) if img.endswith(('.png', '.jpg'))]
header = overlayList[0]
drawColor = (255, 0, 255)

# -------------------- Video Capture --------------------
cap = cv2.VideoCapture(0)
cap.set(3, canvasSize[1])  # Width
cap.set(4, canvasSize[0])  # Height

detector = htm.handDetector(detectionCon=0.8, maxHands=1)
xp, yp = 0, 0
imgCanvas = np.zeros((canvasSize[0], canvasSize[1], 3), np.uint8)

while True:
    success, img = cap.read()
    if not success:
        print("Camera not detected.")
        break

    img = cv2.flip(img, 1)
    img = cv2.resize(img, (canvasSize[1], canvasSize[0]))  # Ensure size is 1280x720
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)

    if len(lmList) != 0:
        x1, y1 = lmList[8][1:]  # Index finger
        x2, y2 = lmList[12][1:]  # Middle finger
        fingers = detector.fingersUp()

        # Selection Mode (2 fingers up)
        if fingers[1] and fingers[2]:
            xp, yp = 0, 0
            if y1 < 125:
                if 100 < x1 < 250:
                    header = overlayList[0]
                    drawColor = (255, 0, 255)
                elif 300 < x1 < 450:
                    header = overlayList[1]
                    drawColor = (0, 255, 0)
                elif 500 < x1 < 650:
                    header = overlayList[2]
                    drawColor = (0, 0, 255)
                elif 700 < x1 < 850:
                    header = overlayList[3]
                    drawColor = (0, 0, 0)  # Eraser
            cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), drawColor, cv2.FILLED)

        # Drawing Mode (only index finger up)
        elif fingers[1] and not fingers[2]:
            cv2.circle(img, (x1, y1), 15, drawColor, cv2.FILLED)
            if xp == 0 and yp == 0:
                xp, yp = x1, y1

            thickness = eraserThickness if drawColor == (0, 0, 0) else brushThickness
            cv2.line(img, (xp, yp), (x1, y1), drawColor, thickness)
            cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, thickness)
            xp, yp = x1, y1

        # Clear Canvas (all fingers up)
        if all(f == 1 for f in fingers):
            imgCanvas = np.zeros_like(imgCanvas)

    # Combine Image and Canvas
    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)

    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, imgCanvas)

    # Overlay Header
    img[0:125, 0:1280] = header
    cv2.putText(img, "Virtual Painter", (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Display
    cv2.imshow("Virtual Painter", img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

