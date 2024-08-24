import cv2 as cv
import random as rd

cap = cv.VideoCapture(0)
face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

color = (0, 255, 0) # rgb
border_color = (0, 0, 255)
circle_r = 30
x_speed = -12
score = 0
score_enabled = True

screen_w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
screen_h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

print('Capture dimensions:', screen_h, screen_w)

line1 = ((int(0.3*screen_w), 0), (int(0.3*screen_w), screen_h))
line2 = ((int(0.7*screen_w), 0), (int(0.7*screen_w), screen_h))

class Obstacle:
    def __init__(self, obstacle_width=60, hole_size=120, color=(255,255,0)):
        global screen_w
        global screen_h
        self.color = color
        self.hole_size = hole_size
        self.width = obstacle_width
        self.superior_pillar = [screen_w, 0, self.width, 0] # x, y, w, h
        self.inferior_pillar = [screen_w, self.hole_size, self.width, screen_h]
    
    def addto_x(self, value:int):
        self.superior_pillar[0] += value
        self.inferior_pillar[0] += value

    def get_x(self):
        return self.superior_pillar[0]
    
    def get_y(self):
        return self.superior_pillar[1]

    def set_x(self, value:int):
        self.superior_pillar[0] = value
        self.inferior_pillar[0] = value

    def set_y(self, value:int):
        self.superior_pillar[3] = value
        self.inferior_pillar[1] = value + self.hole_size

    def randomize_in_right(self):
        self.set_x(screen_w)
        self.set_y(rd.choice(range(0, screen_h-200)))

    def get_pillars(self):
        return [self.superior_pillar, self.inferior_pillar]

    def draw(self, frame):
        p1 = self.superior_pillar
        p2 = self.inferior_pillar
        cv.rectangle(frame, (p1[0], p1[1]), (p1[0]+p1[2], p1[1]+p1[3]), self.color, 5)
        cv.rectangle(frame, (p2[0], p2[1]), (p2[0]+p2[2], p2[1]+p2[3]), self.color, 5)

    def touching(self, circle):
        for pillar in self.get_pillars():
            is_between_x1_and_x2 = circle[0] + circle_r > pillar[0] and (pillar[0] + pillar[2]) > circle[0] - circle_r
            is_between_y1_and_y2 = circle[1] + circle_r > pillar[1] and (pillar[1] + pillar[3]) > circle[1] - circle_r
        
            if is_between_x1_and_x2 and is_between_y1_and_y2:
                return True
        return False

def error(msg:str):
    print("\033[31m" + msg + "\033[m")
    exit(1)

def warning(msg:str):
    print("\033[33m" + msg + "\033[m")

obstacle = Obstacle()

def detect_face(frame):
    global face_cascade, border_color, color, obstacle, score, score_enabled

    face_img = frame.copy()
    face_img = cv.cvtColor(face_img, cv.COLOR_RGB2BGR)

    face_rect = face_cascade.detectMultiScale(face_img, scaleFactor=1.2, minNeighbors=5)
    for (x, y, w, h) in face_rect:
        coord = (x+w//2, y+h//2)

        if obstacle.touching((coord[0], coord[1])):
            color = (255, 0, 0)
            score=0
            score_enabled = False
        else:
            color = (0, 255, 0)

        if coord[0] < obstacle.get_x() + 6 and coord[0] >= obstacle.get_x() - 6:
            score+=1
            score_enabled = False

        if obstacle.get_x() <= line1[0][0]:
            score_enabled = True

        if(coord[0]>line1[0][0] and coord[0]<line2[0][0]):
            cv.circle(face_img, coord, circle_r, color, 5)

    cv.line(face_img, line1[0], line1[1], border_color, 3)
    cv.line(face_img, line2[0], line2[1], border_color, 3)
    obstacle.draw(face_img)
    cv.putText(face_img, str(score), (20, 90), cv.FONT_HERSHEY_COMPLEX, 3, (255, 255, 255), 3)

    return cv.cvtColor(face_img, cv.COLOR_BGR2RGB)

if not cap.isOpened(): error("Unable to open camera")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        warning("Error reading from camera")
        break

    frame_face = detect_face(frame)
    cv.imshow("Flappy Nose", frame_face)

    obstacle.addto_x(x_speed)
    if obstacle.superior_pillar[0] + obstacle.superior_pillar[2] < 0:
        obstacle.randomize_in_right()

    key = cv.waitKey(1) & 0xFF
    
    if key == ord('q'): break
    elif key == ord('p'): cv.imwrite("snapshot.jpg", frame_face)

cap.release()
cv.destroyAllWindows()