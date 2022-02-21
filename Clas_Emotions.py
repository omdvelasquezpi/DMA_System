# _*_ coding:utf-8 _*_
import sys
import dlib  
import numpy as np  
import cv2  
import matplotlib.pyplot as plt
import os
import pickle
import errno
import argparse
import sys
import time
import math
import joblib





class face_dlib():
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, 480)
        self.cnt = 0

    def face(self):
        test_a = 'C:/Users/jnmor/Desktop/ForSteos/face.jpg'
        folder = 'C:/Users/jnmor/Desktop/ForSteos/'

        try:
            os.mkdir(folder)
        except OSError as e:
            if e.errno == errno.EEXIST:
                print('Directory not created.')
            else:
                raise


        # carga el modelo            
        data_filename_memmap = os.path.join(folder,'TrainedModel.sav')
        print(data_filename_memmap)

        #pickle.dump(data_filename_memmap, open("prot2", "w"), protocol=0)
        neighb = joblib.load(data_filename_memmap)
       
        while (self.cap.isOpened()):
            flag, im_rd = self.cap.read()
            k = cv2.waitKey(1)
            img_gray = cv2.cvtColor(im_rd, cv2.COLOR_RGB2GRAY)
            faces = self.detector(img_gray, 0)
            font = cv2.FONT_HERSHEY_SIMPLEX
            if (len(faces) ==1):                
                for k, d in enumerate(faces):
                    #draw red rectangle on the face
                    cv2.rectangle(im_rd, (d.left(), d.top()), (d.right(), d.bottom()), (0, 0, 255))
                    faces = img_gray[d.top():d.bottom(), d.left():d.right()]
                    cropped = img_gray[d.top():d.bottom(), d.left():d.right()]
                    cropped_cropped = cv2.resize(cropped, (48, 48))
                    cv2.imshow("face",cropped_cropped)
                    cv2.imwrite('face.jpg', cropped_cropped)
                    img = plt.imread(test_a)
                    img = np.array(img).reshape(len(img[0])**2)
                    img =  np.array(img).reshape(1,-1)
                    R = neighb.predict(img)

                    main(R[0])
                    #print(R[0])
            else:
                print('Just one face')
                continue                
            # enter "s" key save the image
            if (k == ord('s')):
                self.cnt += 1
                cv2.imwrite("screenshoot" + str(self.cnt) + ".jpg", im_rd)
            # enter "q" to quit
            if (k == ord('q')):
                break
            cv2.imshow("camera", im_rd)
        # free the camera
        self.cap.release()
        # delete the window
        cv2.destroyAllWindows()


def main(R):
    if R == 0:
        print("Angry")
        #time.sleep(3.0)
    elif R == 1:
       
        print("disgust")
        #time.sleep(3.0)
    elif R == 2:
     
        print("fear")
        #time.sleep(3.0)
    elif R == 3:
       
        print("happy")
        #time.sleep(3.0)
    elif R == 4:
     
        print("neutral")
        #time.sleep(3.0)
    elif R == 5:
       
        print("sad")
        #time.sleep(3.0)
    elif R == 6:
       
        print("surprise")
        #time.sleep(3.0)
    else:
        print("IDK")
        #time.sleep(3.0)


if __name__ == "__main__":
    face_d = face_dlib()
    face_d.face()

    


