import cv2
from ultralytics import YOLO
model = YOLO('yolov8n.pt')

cap = cv2.VideoCapture(0)
ret, image = cap.read()

total_x, total_y = image.shape[1]/2, image.shape[0]/2

print(total_x, total_y)


# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    # Read a frame from the camera
    ret, image = cap.read()

    # Check if the frame was read successfully
    if not ret:
        print("Error: Could not read frame.")
        break

    results = model.predict(image, classes = 39)
    for i in range(len(results[0].boxes.xyxy)):
        result = results[0].boxes.xyxy[0]
        cv2.rectangle(image, (int(result[0]), int(result[1])), (int(result[2]), int(result[3])), 0, 4)
        x_mid = (result[0] + result[2]) / 2
        y_mid = (result[1] + result[3]) / 2
        cv2.circle(image, (int(x_mid), int(y_mid)), 1, (0, 255, 0), 5)

        x_deviation = x_mid - total_x
        y_deviation = y_mid - total_y
        # print(result)
        # print(x_deviation, y_deviation)
        print(result[2] - result[0])

    cv2.circle(image, (int(total_x), int(total_y)), 1, (0, 0, 255), 5)


    # Display the frame
    cv2.imshow('Video', image)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break