import cv2
from ultralytics import YOLO
from COCO import coco_classes

OBJECT = 'Person'
TOLERANCE = 20
OBJECT_SIZE = 470

# Style Parameters
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.5
COLOR = (0, 0, 0)
FONT_THICKNESS = 1

def take_action(x, size, size_desired, x_tolerance=TOLERANCE, size_tolerance=TOLERANCE):
    text = ""
    if abs(size - size_desired) > size_tolerance:
        text += "Move Backward " if size > size_desired else "Move Forward "
    if abs(x) > x_tolerance:
        text += "Rotate Left" if x < 0 else "Rotate Right"
    return text if text else "None"

if __name__ == '__main__':
    model = YOLO('yolov8n.pt')

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Use cv2.CAP_DSHOW for Windows, remove otherwise

    if not cap.isOpened():
        print("Error: Could not open camera.")
        exit()

    center_x, center_y = cap.get(3) / 2, cap.get(4) / 2  # Use cap.get() to get frame dimensions

    while True:
        ret, image = cap.read()

        if not ret:
            print("Error: Could not read frame.")
            break

        results = model.predict(image, classes=[coco_classes[OBJECT]])

        if len(results[0].boxes.xyxy) > 0:
            result = results[0].boxes.xyxy[0]
            x_mid = (result[0] + result[2]) / 2
            y_mid = (result[1] + result[3]) / 2
            x_deviation = x_mid - center_x
            y_deviation = center_y - y_mid
            size = result[2] - result[0]
            action = take_action(x_deviation, size, OBJECT_SIZE)
            print("Action", action, f"X: {int(x_deviation)}", f"Y: {int(y_deviation)}", f"Size: {int(size)}")

            # Additional display code can be added here if needed

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
