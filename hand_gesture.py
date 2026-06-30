import cv2
import numpy as np

# Open webcam
cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    if not ret:
        print("Camera not working")
        break

    # Flip image
    frame = cv2.flip(frame, 1)

    # Define hand area
    roi = frame[100:300, 100:300]

    # Convert to HSV
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Skin color range
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)

    # Mask
    mask = cv2.inRange(hsv, lower_skin, upper_skin)

    # Blur
    mask = cv2.GaussianBlur(mask, (5,5), 0)

    # Find contours
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # Draw largest contour
    if len(contours) > 0:

        max_contour = max(contours, key=cv2.contourArea)

        cv2.drawContours(roi, [max_contour], -1, (0,255,0), 2)

        cv2.putText(
            frame,
            "Hand Detected",
            (50,50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,255,0),
            2
        )

    # Draw rectangle
    cv2.rectangle(frame, (100,100), (300,300), (255,0,0), 2)

    # Show windows
    cv2.imshow("Hand Gesture", frame)
    cv2.imshow("Mask", mask)

    # Exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()