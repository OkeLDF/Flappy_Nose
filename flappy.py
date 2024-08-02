import cv2 as cv

cap = cv.VideoCapture(0)
face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

color = (255, 0, 0) # rgb

def error(msg:str):
    print("\033[31m" + msg + "\033[m")
    exit(1)

def warning(msg:str):
    print("\033[33m" + msg + "\033[m")

def detect_face(frame):
    global face_cascade
    face_img = frame.copy()
    face_img = cv.cvtColor(face_img, cv.COLOR_RGB2BGR)

    face_rect = face_cascade.detectMultiScale(face_img, scaleFactor=1.2, minNeighbors=5)
    for (x,y, w, h) in face_rect:
        r = 30
        cv.circle(face_img, (x+w//2, y+h//2), r, color, 2)

    return cv.cvtColor(face_img, cv.COLOR_BGR2RGB)

if not cap.isOpened(): error("Unable to open camera")

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        warning("Error reading from camera")
        break

    frame_face = detect_face(frame)
    cv.imshow("Webcam Feed", frame_face)
    
    key = cv.waitKey(1) & 0xFF
    
    if key == ord('q'): break
    elif key == ord('p'): cv.imwrite("snapshot.jpg", frame)

cap.release()
cv.destroyAllWindows()