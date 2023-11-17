from COCO import coco_classes
import cv2
from ultralytics import YOLO

OBJECT = 'Person'
TOLERENCE = 20
OBJECT_SIZE = 470

# Style Parameters
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.5
COLOR = (0, 0, 0)
FONT_THICKNESS = 1
RADIUS = 1
THICKNESS = 5

def takeAction(x, size, size_desired, x_tolerence=TOLERENCE, size_tolerence=TOLERENCE):
    text = ""
    if abs(size - size_desired) > size_tolerence:
        if(size > size_desired):
            text += "Move Backward "
        elif(size < size_desired):
            text += "Move Forward "
    if abs(x) > x_tolerence:
        if x < 0:
            text += "Rotate Left"
        elif x > 0:
            text += "Rotate Right"
    if text == "":
        text = "None"
    return text

if __name__ == '__main__':

    model = YOLO('yolov8n.pt')

    cap = cv2.VideoCapture(0)
    ret, image = cap.read()

    centre_x, centre_y = image.shape[1]/2, image.shape[0]/2

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
        
        results = model.predict(image, classes = [coco_classes[OBJECT]])

        # for i in range(len(results[0].boxes.xyxy)):
        # result = results[0].boxes.xyxy[i]
        
        if len(results[0].boxes.xyxy > 0):
            objectFound = True
        else:
            objectFound = False

        # If object is present
        if objectFound == True:    
            result = results[0].boxes.xyxy[0]

            x_mid = (result[0] + result[2]) / 2
            y_mid = (result[1] + result[3]) / 2

            x_deviation = x_mid - centre_x
            y_deviation = centre_y - y_mid
            size = result[2] - result[0]
            action = takeAction(x_deviation, size, OBJECT_SIZE)

            # Bounding Box
        #     cv2.rectangle(image, (int(result[0]), int(result[1])), (int(result[2]), int(result[3])), 0, 4)

        #     # Object Centre
        #     cv2.circle(image, (int(x_mid), int(y_mid)), RADIUS, (0, 255, 0), THICKNESS)

        #     # Texts
        #     cv2.rectangle(image, (0, int(centre_y)*2-30), (int(centre_x)*2, int(centre_y)*2), (255, 255, 255), -1)
        #     cv2.putText(image, f"X: {int(x_deviation)}", (0, int(centre_y)*2-5), FONT, FONT_SCALE, COLOR, FONT_THICKNESS) 
        #     cv2.putText(image, f"Y: {int(y_deviation)}", (100, int(centre_y)*2-5), FONT, FONT_SCALE, COLOR, FONT_THICKNESS) 
        #     cv2.putText(image, f"Size: {int(size)}", (200, int(centre_y)*2-5), FONT, FONT_SCALE, COLOR, FONT_THICKNESS) 
        #     cv2.putText(image, "Action: " + action, (int(centre_x), int(centre_y)*2-5), FONT, FONT_SCALE, COLOR, FONT_THICKNESS) 
        
        # # Frame centre
        # cv2.circle(image, (int(centre_x), int(centre_y)), RADIUS, (0, 0, 255), THICKNESS)
        
        # # Texts
        # if objectFound == False:
        #     cv2.rectangle(image, (0, int(centre_y)*2-30), (int(centre_x)*2, int(centre_y)*2), (255, 255, 255), -1)
        #     cv2.putText(image, f"X: -", (0, int(centre_y)*2-5), FONT, FONT_SCALE, COLOR, FONT_THICKNESS) 
        #     cv2.putText(image, f"Y: -", (100, int(centre_y)*2-5), FONT, FONT_SCALE, COLOR, FONT_THICKNESS) 
        #     cv2.putText(image, f"Size: -", (200, int(centre_y)*2-5), FONT, FONT_SCALE, COLOR, FONT_THICKNESS) 
        #     cv2.putText(image, "Action: -", (int(centre_x), int(centre_y)*2-5), FONT, FONT_SCALE, COLOR, FONT_THICKNESS) 
        

        # # Display the frame
        # cv2.imshow('Video', image)

        # Press 'q' to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break