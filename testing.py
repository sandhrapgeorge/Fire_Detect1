import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from keras.preprocessing import image
from keras.models import load_model
import random
import imutils
import sys
import glob
import smtplib
import playsound
import threading
Alarm_Status = False
Email_Status = False

class YOLO:

    def __init__(self, config, model, labels, size=416, confidence=0.5, threshold=0.3):
        self.confidence = confidence
        self.threshold = threshold
        self.size = size
        self.labels = labels
        self.net = cv2.dnn.readNetFromDarknet(config, model)

    def inference(self, image):
        ih, iw = image.shape[:2]

        ln = self.net.getLayerNames()
        ln = [ln[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (self.size, self.size), swapRB=True, crop=False)
        self.net.setInput(blob)
        start = time.time()
        layerOutputs = self.net.forward(ln)
        end = time.time()
        inference_time = end - start

        boxes = []
        confidences = []
        classIDs = []
        for output in layerOutputs:
            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]
                if confidence > self.confidence:
                    box = detection[0:4] * np.array([iw, ih, iw, ih])
                    (centerX, centerY, width, height) = box.astype("int")
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        idxs = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence, self.threshold)

        results = []
        if len(idxs) > 0:
            for i in idxs.flatten():
                x, y = (boxes[i][0], boxes[i][1])
                w, h = (boxes[i][2], boxes[i][3])
                id = classIDs[i]
                confidence = confidences[i]

                print(self.labels)
                results.append((id, self.labels[id], confidence, x, y, w, h))

        return iw, ih, inference_time, results


class Foo(object):
    counter = 0

    def __call__(self):
        Foo.counter += 1
        return (Foo.counter)


def email_alert():
    import getpass
    import smtplib
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.ehlo()

    server.starttls()
    username = 'sandhrageorge123@gmail.com'
    passwd = 'hancepaul'
    server.login(username, passwd)

    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText
    from email.mime.application import MIMEApplication
    from datetime import datetime

    msg = MIMEMultipart()
    msg['from'] = username
    msg['to'] = username
    msg['subject'] = "Fire Images"
    text = "Found Fire, have a look at sample images"
    msg.attach(MIMEText(text))
    F = glob.glob("detected-images/*")

    count = 0
    for i in F:
        with open(i, 'rb') as f:
            part = MIMEApplication(f.read())
            part.add_header('content-Disposition', 'attachment', filename='{}.jpg'.format(count + 1))
            msg.attach(part)
    server.sendmail(username, username, msg.as_string())


def play_alarm_sound_function():
    while True:
        playsound.playsound('Alarm Sound.mp3', True)


cap = cv2.VideoCapture('input2.mp4')
img_counter = 0
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output-1.avi', fourcc, 20.0, (640, 480))
yolo = YOLO("yolov3_testing.cfg", "yolov3_training_last.weights", ["Fire"])
foo = Foo()

while True:
    img_counter += 1
    frames = []
    print("frame: ", img_counter)

    for i in range(0, 16):
        grabbed, frame = cap.read()
        if not grabbed:
            print("[Info] No frame read from stream - exiting")
            out.release()
            sys.exit(0)

        frame = imutils.resize(frame, width=400)
        frames.append(frame)

    width, height, inference_time, results = yolo.inference(frames[-1])

    for detection in results:
        id, name, confidence, x, y, w, h = detection
        color = (0, 255, 255)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, 'Fire', (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.imwrite('detected-images/{}.jpg'.format(img_counter), frame)
        if (foo() == 4):
            print("Found Fire more than 3 frames, Raising Email-Alert")
            if Email_Status == False:
                threading.Thread(target=email_alert).start()
                Email_Status = True
            if Alarm_Status == False:
                threading.Thread(target=play_alarm_sound_function).start()
                Alarm_Status = True
    out.write(frame)
    prv = cv2.resize(frame, (500, 500))
    cv2.imshow("preview", prv)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()