from COCO import coco_classes
# from ultralytics import YOLO
import numpy as np
from picamera2 import PiCamera2
from picamera2.array import PiRGBArray
import time
import cv2

OBJECT = 'Person'
TOLERANCE = 20
OBJECT_SIZE = 470

# Style Parameters
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.5
COLOR = (0, 0, 0)
FONT_THICKNESS = 1

def takeAction(x, size, size_desired, x_tolerance=TOLERANCE, size_tolerance=TOLERANCE):
    text = ""
    if abs(size - size_desired) > size_tolerance:
        if size > size_desired:
            text += "Move Backward "
        elif size < size_desired:
            text += "Move Forward "
    if abs(x) > x_tolerance:
        if x < 0:
            text += "Rotate Left"
        elif x > 0:
            text += "Rotate Right"
    if text == "":
        text = "None"
    return text

if __name__ == '__main__':
    # model = YOLO('yolov8n.pt')

    # Initialize PiCamera
    camera = PiCamera2()
    camera.resolution = (640, 480)
    camera.framerate = 30
    raw_capture = PiRGBArray(camera, size=(640, 480))

    # Warm-up time for the camera
    time.sleep(2)

    for frame in camera.capture_continuous(raw_capture, format="bgr", use_video_port=True):
        image = frame.array

        # results = model.predict(image, classes=[coco_classes[OBJECT]])

        # if len(results[0].boxes.xyxy) > 0:
        #     objectFound = True
        # else:
        #     objectFound = False

        # # If object is present
        # if objectFound:
        #     result = results[0].boxes.xyxy[0]

        #     x_mid = (result[0] + result[2]) / 2
        #     y_mid = (result[1] + result[3]) / 2

        #     x_deviation = x_mid - (image.shape[1] / 2)
        #     y_deviation = (image.shape[0] / 2) - y_mid
        #     size = result[2] - result[0]
        #     action = takeAction(x_deviation, size, OBJECT_SIZE)
        #     print("Action", action, f"X: {int(x_deviation)}", f"Y: {int(y_deviation)}", f"Size: {int(size)}")

        #     # Display the result on the image
        #     cv2.putText(image, f"X: {int(x_deviation)}", (0, image.shape[0] - 5), FONT, FONT_SCALE, COLOR, FONT_THICKNESS)
        #     cv2.putText(image, f"Y: {int(y_deviation)}", (100, image.shape[0] - 5), FONT, FONT_SCALE, COLOR, FONT_THICKNESS)
        #     cv2.putText(image, f"Size: {int(size)}", (200, image.shape[0] - 5), FONT, FONT_SCALE, COLOR, FONT_THICKNESS)
        #     cv2.putText(image, "Action: " + action, (int(image.shape[1] / 2), image.shape[0] - 5), FONT, FONT_SCALE, COLOR, FONT_THICKNESS)

        # # Display the image
        # cv2.imshow('Camera Feed', image)

        # Clear the stream in preparation for the next frame
        raw_capture.truncate(0)

        # Press 'q' to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera
    camera.close()
    cv2.destroyAllWindows()
