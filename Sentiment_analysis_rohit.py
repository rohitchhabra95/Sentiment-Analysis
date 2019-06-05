from __future__ import division 
from imutils import face_utils
import imutils
import dlib
#from gtts import gTTS
import os
import subprocess
import numpy as np
from math import sqrt
from sklearn import neighbors
from os import listdir
from os.path import isdir, join, isfile, splitext
import pickle
from PIL import Image, ImageFont, ImageDraw, ImageEnhance
import face_recognition
from face_recognition import face_locations
#from face_recognition.cli import image_files_in_folder
from face_recognition.face_recognition_cli import image_files_in_folder
import cv2
from keras.models import model_from_json
mqtt_flag=True
try:
    from mqtt import *
except:
    mqtt_flag=False
    print("MQTT is not imported")
    pass
#from Music_gui import music_gui
import speech_recognition as sr
from keras import optimizers
import time
import random
from keras.models import model_from_json
from keras.preprocessing import image




ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg','JPEG'}
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
Nm_lst=['Rupender','Satyam','Pragati','Rohit']
emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
Sng_lst_no=[0,1,3,4,5,6]
#r=sr.Recognizer()
"""def spch_rcg():
    spch_rslt=None
    device_index=1
    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source,duration=1)
        print('speak')
        
        audio=r.listen(source, timeout=3)
        
            
    try:
        spch_rslt=r.recognize_google(audio)
    
        return spch_rslt
    except sr.UnknownValueError:
        print('Could not understand you')
        publish_server('CODE=Could not understand you')
        time.sleep(3)
    except sr.RequestError as e:
        print('Request error')
        if mqtt_flag==True:
            publish_server('CODE=Could not request result because of network issue')
            time.sleep(3)
    return spch_rslt
    """
def train(train_dir, model_save_path = "", n_neighbors = None, knn_algo = 'ball_tree', verbose=False):

    X = []
    y = []
    for class_dir in listdir(train_dir):
        
        if not isdir(join(train_dir, class_dir)):
            
            continue
        
        for img_path in image_files_in_folder(join(train_dir, class_dir)):
            print('mmmmm',img_path)
            image = face_recognition.load_image_file(img_path)
            faces_bboxes = face_locations(image)
            if len(faces_bboxes) != 1:
                if verbose:
                    print("image {} not fit for training: {}".format(img_path, "didn't find a face" if len(faces_bboxes) < 1 else "found more than one face"))
                continue
            X.append(face_recognition.face_encodings(image, known_face_locations=faces_bboxes)[0])
            y.append(int(class_dir))
    print(y)        

    if n_neighbors is None:
        n_neighbors = int(round(sqrt(len(X))))
        if verbose:
            print("Chose n_neighbors automatically as:", n_neighbors)
    X,y=np.asarray(X),np.asarray(y)
    knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance')
    knn_clf.fit(X, y)

    if model_save_path != "":
        with open(model_save_path, 'wb') as f:
            pickle.dump(knn_clf, f)
    return knn_clf

def predict(frame,dist_thrsh,knn_clf = None, model_save_path =""):
    #global count
    DIST_THRESH=dist_thrsh
    if knn_clf is None and model_save_path == "":
        raise Exception("must supply knn classifier either thourgh knn_clf or model_save_path")

    if knn_clf is None:
        with open(model_save_path, 'rb') as f:
            knn_clf = pickle.load(f)
    try:
        X_faces_loc=face_recognition.face_locations(frame)
        #print(X_faces_loc)
        faces_encodings = face_recognition.face_encodings(frame, known_face_locations=X_faces_loc)


        closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)

        is_recognized = [closest_distances[0][i][0] <= DIST_THRESH for i in range(len(X_faces_loc))]
    except:
        pass

    # predict classes and cull classifications that are not with high confidence
    if len(X_faces_loc)!=0:
        return [(pred, loc) if rec else ("Unknown", loc) for pred, loc, rec in zip(knn_clf.predict(faces_encodings), X_faces_loc, is_recognized)]
    else:
        return None
def draw_preds(preds,frame):
    source_img = Image.fromarray(frame.astype('uint8'), 'RGB')
    draw = ImageDraw.Draw(source_img)
    for pred in preds:
        loc = pred[1]
        name = pred[0]
        #(top, right, bottom, left) => (left,top,right,bottom)
        if name=="Unknown":
            draw.rectangle(((loc[3], loc[0]), (loc[1],loc[2])), outline="blue")
            draw.rectangle(((loc[3]+1, loc[0]+1), (loc[1]-1, loc[2]-1)), outline="blue")
            draw.rectangle(((loc[3]+2, loc[0]+2), (loc[1]-2, loc[2]-2)), outline="blue")
            #draw.rectangle(((loc[3] + 3, loc[0] + 3), (loc[1] -3, loc[2] - 3)), outline="red")
            #draw.rectangle(((loc[3], loc[0] -35), (loc[1] , loc[0] )), outline="red",fill=(255,0,0,255))
            #draw.text((loc[3], loc[0] - 30), name, font=ImageFont.truetype('Pillow/Tests/fonts/FreeMono.ttf', 30))
            
        else:
            draw.rectangle(((loc[3], loc[0]), (loc[1], loc[2])), outline="green")
            draw.rectangle(((loc[3] + 1, loc[0] + 1), (loc[1] - 1, loc[2] - 1)), outline="green")
            draw.rectangle(((loc[3] + 2, loc[0] + 2), (loc[1] - 2, loc[2] - 2)), outline="green")
            draw.rectangle(((loc[3] + 3, loc[0] + 3), (loc[1] - 3, loc[2] - 3)), outline="green")
            draw.rectangle(((loc[3], loc[0] - 35), (loc[1], loc[0])), outline="green", fill=(0,255,0,255))
            #draw.text((loc[3], loc[0] - 30), name, font=ImageFont.truetype('Pillow/Tests/fonts/FreeMono.ttf', 30))
            dist=loc[1]-loc[3]
    return source_img
	
if __name__ == "__main__":
    knn_clf = train("face_images/train",model_save_path="FaceModel.p")
    
    print('Start')
    K = True
    model = model_from_json(open("facial_expression_model_structure.json", "r").read())
    model.load_weights('facial_expression_model_weights.h5')
    face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    emotion_lst=[]
    cnt=0
    frm=0
    with open("FaceModel.p", 'rb') as f:
        face_model = pickle.load(f)
    cap=cv2.VideoCapture(0)
    #print(cap)
    while True:
        
        ok,frame1=cap.read()
        #print('mmm')
        image=frame1
        frame=frame1
        if ok==False:
            print('Break')
            break
        frm=frm+1
        if K:
            if frm%2==0:
                #print(K)
                frame1 = frame1[:, :, ::-1]
                preds = predict(frame1, knn_clf=face_model,dist_thrsh=0.5)
                if preds==None:
                    continue
                print(preds[0][0])
                frame=draw_preds(preds, frame1)
                frame=np.array(frame)
                frame=frame[:, :, ::-1]
                if preds[0][0]!='Unknown':
                    K=False
                    try:
                        publish_server('CODE='+'Welcome, '+str(Nm_lst[preds[0][0]])+' I recognized you.')
                        time.sleep(3)
                    except:
                        pass
                else:
                    try:
                        publish_server('CODE= I did not recognize you.')
                        time.sleep(4)
                    except:
                        pass
        
        #print('Expression start')
        else:
            print('emotion')
            img=image
            gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            faces=face_cascade.detectMultiScale(gray,1.3,5)
            cnt=cnt+1
            if len(faces)>=1:
                x,y,w,h=faces[0][0],faces[0][1],faces[0][2],faces[0][3]
                if w*h>14000:
                    detected_face=img[int(y):int(y+h),int(x):int(x+w)]
                    detected_face=cv2.cvtColor(detected_face,cv2.COLOR_BGR2GRAY)
                    img_pixels=cv2.resize(detected_face,(48,48))
                    #img_pixels=image.img_to_array(detected_face)
                    #img_pixels=np.expand_dims(img_pixels,axis=0)
                    img_pixels=img_pixels.astype(float)
                    #print(img_pixels[0])
                    img_pixels /=255
                    img_pixels = np.reshape(img_pixels, (48, 48, 1))
                    img_pixels=np.expand_dims(img_pixels,axis=0)
                    
                    prediction=model.predict(img_pixels)
                    max_index=np.argmax(prediction[0])
                    #emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
                    emotion = emotions[max_index]
                    if emotion=="fear" or emotion=="surprise":
                        emotion=random.choice(['fear','surprise','angry'])
                        emotion_lst.append(emotions.index(emotion))
                        print(emotion)
                    elif emotion=="neutral":
                        emotion=random.choice(['neutral','disgust'])
                        emotion_lst.append(emotions.index(emotion))
                    else:
                        pass
                    cv2.putText(img, emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            if len(emotion_lst)>7 and cnt>10:
                mx=max(emotion_lst,key=emotion_lst.count)
                print('maximum_emotion',mx)
                try:
                    publish_server('CODE= Hey you are'+str(emotions[mx])+','+' Would you like to listen some music?')
                    time.sleep(3)
                except:
                    pass
                emotion_lst.clear()
                cnt=0
                if mqtt_flag==True:
                    """speech_result=spch_rcg()
                    if speech_result!=None:
                        speech_result=str(speech_result)
                        #print(speech_result)
                        lst=speech_result.split()
                        print(lst)
                        if 'yes' in lst or 'yeah' in lst or 'yup' in lst or 'please' in lst or 'sure' in lst or 'haan' in lst or 'han' in lst or 'ha' in lst or 'ok' in lst or 'go' in lst or 'ahead' in lst:
                            print('')
                            try:
                                music_gui(mx)
                            except:
                                pass
                        else:
                            pass """
            frame=img
            
        
        cv2.imshow("output",frame)
        k=cv2.waitKey(1)
        if k==27:
            break
    cap.release()
    cv2.destroyAllWindows()
    