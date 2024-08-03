import cv2 as cv

cap = cv.VideoCapture(0)
face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

color = (0, 255, 0) # rgb
border_color = (0, 0, 255)

def error(msg:str):
    print("\033[31m" + msg + "\033[m")
    exit(1)

def warning(msg:str):
    print("\033[33m" + msg + "\033[m")

def detect_face(frame):
    global face_cascade
    face_img = frame.copy()
    face_img = cv.cvtColor(face_img, cv.COLOR_RGB2BGR)

    fw = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

    line1 = ((int(0.3*fw), 0), (int(0.3*fw), fh))
    line2 = ((int(0.7*fw), 0), (int(0.7*fw), fh))

    cv.line(face_img, line1[0], line1[1], border_color, 3)
    cv.line(face_img, line2[0], line2[1], border_color, 3)

    face_rect = face_cascade.detectMultiScale(face_img, scaleFactor=1.2, minNeighbors=5)
    for (x,y, w, h) in face_rect:
        r = 30
        coord = (x+w//2, y+h//2)
        if(coord[0]>line1[0][0] and coord[0]<line2[0][0]):
            cv.circle(face_img, coord, r, color, 5)

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
    elif key == ord('p'): cv.imwrite("snapshot.jpg", frame_face)

cap.release()
cv.destroyAllWindows()