import tkinter as tk
from tkinter import *
 
from PIL import Image, ImageTk

import os
import numpy as np
import cv2
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D
from keras.optimizers import Adam
from keras.layers import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator

emotion_model = Sequential()

emotion_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
emotion_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))

emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))

emotion_model.add(Flatten())
emotion_model.add(Dense(1024, activation='relu'))
emotion_model.add(Dropout(0.5))
emotion_model.add(Dense(7, activation='softmax'))
emotion_model.load_weights('C:/Users/Kamtam Srichandana/Desktop/SMP NEW/model.h5')

cv2.ocl.setUseOpenCL(False)

emotion_dict = {0: "Angry", 
1: "Disgusted", 
2: "Fearful",
3: "Happy",
4: "Neutral", 
5: "Sad", 
6: "Surprised"}


emoji_dist={0:"C:/Users/Kamtam Srichandana/Desktop/SMP NEW/Emojis/angry.png",
1:"C:/Users/Kamtam Srichandana/Desktop/SMP NEW/Emojis/disgusted.png",
2:"C:/Users/Kamtam Srichandana/Desktop/SMP NEW/Emojis/fearful.png",
3:"C:/Users/Kamtam Srichandana/Desktop/SMP NEW//Emojis/happy.png",
4:"C:/Users/Kamtam Srichandana/Desktop/SMP NEW/Emojis/neutral.png",
5:"C:/Users/Kamtam Srichandana/Desktop/SMP NEW/Emojis/sad.png",
6:"C:/Users/Kamtam Srichandana/Desktop/SMP NEW/Emojis/surprised.png"}

global last_frame1
last_frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
global cap1
show_text=[0]

# Function to get face captured and recognize emotion
def Capture_Image():
    global cap1
    cap1 = cv2.VideoCapture(0)
    if not cap1.isOpened():
        print("cant open the camera1")
    flag1, frame1 = cap1.read()
    frame1 = cv2.resize(frame1,(500,500))
    # It will detect the face in the video and bound it with a rectangular box
    bound_box = cv2.CascadeClassifier('C:/Users/Kamtam Srichandana/Desktop/SMP NEW/haarcascade_frontalface_default.xml')
    gray_frame = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    n_faces = bound_box.detectMultiScale(gray_frame,scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in n_faces:
        cv2.rectangle(frame1, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
        roi_frame = gray_frame[y:y + h, x:x + w]
        crop_img = np.expand_dims(np.expand_dims(cv2.resize(roi_frame, (48, 48)), -1), 0)
        prediction = emotion_model.predict(crop_img)
        maxindex = int(np.argmax(prediction))
        print(emotion_dict[maxindex])
        cv2.putText(frame1, emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        show_text[0]=maxindex

    if flag1 is None:
        print ("Error!")

    elif flag1:
        global last_frame1
        last_frame1 = frame1.copy()
        pic = cv2.cvtColor(last_frame1, cv2.COLOR_BGR2RGB) #to store the image
        img = Image.fromarray(pic)
        imgtk = ImageTk.PhotoImage(image=img)
        label1.imgtk = imgtk
        label1.configure(image=imgtk)
        label1.after(10, Capture_Image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        exit()

# Function for showing Emoji According to Facial Expression
def Get_Emoji():
    frame2=cv2.imread(emoji_dist[show_text[0]])
    pic2=cv2.cvtColor(frame2,cv2.COLOR_BGR2RGB)
    img2=Image.fromarray(frame2)
    imgtk2=ImageTk.PhotoImage(image=img2)
    label2.imgtk2=imgtk2
    label3.configure(text=emotion_dict[show_text[0]],font=('arial',45,'bold'))
    label2.configure(image=imgtk2)
    label2.after(10, Get_Emoji)

# GUI Window to show captured image with emoji
if __name__ == '__main__':
    root = Tk()
    root.title("Emojify")
    root.configure(background='black')
    root.geometry("1200x800+10+20")
    # a = Label(root, text='SUMMER INTERNSHIP PROJECT',bg='black')
    # a.pack()
    heading2=Label(root,text="Emojify",pady=20, font=('arial',45,'bold'),bg='black',fg='#CDCDCD')#to label the output
    heading2.pack() 
    label1=Label(master=root,font=("Times",30,"bold"), bg='red')
    label1.pack(side=LEFT, padx=50,pady=15)
    label2=Label(root,font=("Times",30,"bold"), bg='blue')
    label2.pack(side=RIGHT, padx=50,pady=15)
    label3=Label(root)
    label3.pack()
    b=Button(root,text="QUIT",command=root.destroy).pack(side=BOTTOM)
    #b1=Button(root,text="CAPTURE").pack(side=BOTTOM)    
    Capture_Image()
    #Get_Emoji()
    root.mainloop()