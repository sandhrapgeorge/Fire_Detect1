from tkinter import *
import tkinter.messagebox
from datetime import datetime
from PIL import Image, ImageTk
from tkinter.font import Font
import sys
from datetime import timedelta
from tkinter import filedialog
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
dateval = ""
##############
ent_fn, ent_ln, ent_age, ent_email, ent_ph, e_gender, txt_addr, ent_user, ent_pswd = 0, 0, 0, 0, 0, 0, 0, 0, 0

import calendar
import datetime

try:
    import Tkinter
    import tkFont
    import ttk

    from Tkconstants import CENTER, LEFT, N, E, W, S
    from Tkinter import StringVar
except ImportError:  # py3k
    import tkinter as Tkinter
    import tkinter.font as tkFont
    import tkinter.ttk as ttk

    from tkinter.constants import CENTER, LEFT, N, E, W, S
    from tkinter import StringVar
from tkinter import filedialog

##############

##myFonthead = Font(family="Times New Roman", size=12)
top = 0

# _________________________________________________SAVE_______________________________________________________
new_top = 0
txt = ""
# _________________________________________________function_login_______________________________________________________
import cv2
import numpy as np
import glob
from keras.preprocessing import image as image_utils
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.models import Sequential, load_model
from PIL import Image, ImageTk
import requests
from io import BytesIO
import os
import smtplib
from twilio.rest import Client
from tkinter import Tk, Label, Canvas, NW, Entry, Button

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
        playsound.playsound('AlarmSound.mp3', True)


def analyzeacc():
    img_width, img_height = 128, 128
    cntt = 0

    filename = filedialog.askopenfilename(initialdir="C:/Users/ACER/PycharmProjects/Fire_detect1/inputs",
                                          title="choose your file")
    cap = cv2.VideoCapture(filename)
    img_counter = 0
    Alarm_Status = False
    Email_Status = False
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output-1.avi', fourcc, 20.0, (640, 480))
    yolo = YOLO("yolov3_testing.cfg", "yolov3_training_last.weights", ["Fire"])

    image_names1 = os.listdir("detected-images")
    for i in image_names1:
        image_path = os.path.join("detected-images", i)
        os.remove(image_path)

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


def back():
    global top
    top.withdraw()
    login()


def close_window(root):
    root.destroy()


def login():
    t = Tk()
    t.withdraw()
    global new_top
    new_top = Toplevel()
    myFont = Font(family="Times New Roman", size=10)
    image = Image.open("cover.jpg")
    image = image.resize((850, 650), Image.ANTIALIAS)  ## The (550, 250) is (height, width)
    pic = ImageTk.PhotoImage(image)
    lbl_reg = Label(new_top, image=pic)
    lbl_reg.place(x=0, y=0)
    new_top.config(bg="#ff8201")
    new_top.minsize(850, 650)
    log_name = StringVar()
    log_pass = StringVar()
    un = log_name.get()
    ps = log_pass.get()
    print("un", un, "ps", ps)
    print("..*..")
    ##
    ##    l1=Label(new_top,text="EYE HOSPITAL",bg="white",font=myFonthead)
    ##    l1.place(x=52,y=20)
    ##
    usr_nm = Label(new_top, text="USER NAME", font=myFont)
    usr_nm.place(x=300, y=200)
    usr_nm_ent = Entry(new_top, textvariable=log_name, bg="white", width=15, font=myFont)
    usr_nm_ent.place(x=435, y=200)

    usr_pass = Label(new_top, text="PASSWORD", font=myFont)
    usr_pass.place(x=300, y=240)
    usr_pass_ent = Entry(new_top, show="*", textvariable=log_pass, bg="white", width=15, font=myFont)
    usr_pass_ent.place(x=435, y=240)

    btn_login = Button(new_top, text="LOGIN", bg="white", fg="black", font=myFont, relief=RIDGE,
                       command=lambda: check_tb(log_name.get(), log_pass.get()))
    btn_login.place(x=400, y=300)

    new_top.mainloop()


# _________________________________________________function_check_tb_____________________________________________________
def user(username):
    global top, txt
    print('Home page loaded')
    top = Toplevel()
    top.geometry("550x150")
    top.title("FORM")
    top.minsize(850, 750)
    top.config(bg="#ff8201")

    image = Image.open("home.jpg")
    image = image.resize((850, 750), Image.ANTIALIAS)  ## The (550, 250) is (height, width)
    pic = ImageTk.PhotoImage(image)
    lbl_reg = Label(top, image=pic)
    lbl_reg.place(x=0, y=0)

    ''' Button 1'''
    myFont = Font(family="Times New Roman", size=10)

    btn2 = Button(top, text='ANALYZE FIRE', bg="light green", font=myFont, height=2, width=20,
                  command=lambda: analyzeacc())
    btn2.place(x=130, y=300)

    btn4 = Button(top, text='LOGOUT', bg="light green", font=myFont, height=2, width=20,
                  command=lambda: back())
    btn4.place(x=130, y=380)
    top.mainloop()


def check_tb(un, ps):
    print(":", un, ":", ps)
    if un == "admin" and ps == "admin":
        print("sucessfull")
        tkinter.messagebox.showinfo("login", "Successful")
        global new_top
        new_top.withdraw()
        fname = un
        user(fname)
    else:
        print("invalid username/password")
        tkinter.messagebox.showinfo("login", "invalid username/password")


login()