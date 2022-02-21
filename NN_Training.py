import os
import numpy as np
import skimage.feature
import matplotlib.pyplot as plt
from sklearn.metrics import (confusion_matrix, classification_report,accuracy_score)
import seaborn as sns
from matplotlib.colors import ListedColormap
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (StandardScaler, MinMaxScaler)
from sklearn.neighbors import (KNeighborsClassifier) 
#NeighborhoodComponentsAnalysis)
from sklearn.pipeline import Pipeline
import joblib
import pickle

train_path = "C:/Users/jnmor/Documents/ROB311/TP4/TP4/train/"
test_path = "C:/Users/jnmor/Documents/ROB311/TP4/TP4/test/"
folders_train = os.listdir(train_path)
folders_test = os.listdir(test_path)
print(folders_train)
print(folders_test)


imgs_train=[] #save the train imamges
lable_list_train=[] # save the train emotion lable  [0 'angry', 1'disgust', 2'fear', 3'happy', 4'neutral', 5'sad', 6'surprise']

for i in range(len(folders_train)):
    path_emotion=train_path+folders_train[i]
    filenames=sorted(os.listdir(path_emotion))      
    #print("..................")
    #print(filenames)
    for j in range(len(filenames)):
        lable_list_train.append(i)
        img=plt.imread(path_emotion+"/"+filenames[j])
        imgs_train.append(img)
# now = datetime.now()
# current_time = now.strftime("%H:%M:%S")
# print("Current Time =", current_time)

imgs_test=[] #save the test imamges
lable_list_test=[] # save the test emotion lable  [0 'angry', 1'disgust', 2'fear', 3'happy', 4'neutral', 5'sad', 6'surprise']

for i in range(len(folders_test)):
    path_emotion=test_path+folders_test[i]
    filenames=sorted(os.listdir(path_emotion))      
    #print("..................")
    #print(filenames)
    for j in range(len(filenames)):
        lable_list_test.append(i)
        img=plt.imread(path_emotion+"/"+filenames[j])
        imgs_test.append(img)
# now = datetime.now()
# current_time = now.strftime("%H:%M:%S")
# print("Current Time =", current_time)


imgs_train_a = np.array(imgs_train).reshape((len(imgs_train), len(imgs_train[0])**2))
lable_list_train_a = np.array(lable_list_train)
imgs_test_a = np.array(imgs_test).reshape((len(imgs_test), len(imgs_test[0])**2))
lable_list_test_a = np.array(lable_list_test)


imgs_np_train=np.array(imgs_train)
imgs_np_train.shape

print(imgs_np_train.shape)

imgs_np_test=np.array(imgs_test)
imgs_np_test.shape

print(imgs_np_test.shape)


neigh = KNeighborsClassifier(n_neighbors=1)
neigh.fit(imgs_train_a,lable_list_train_a)

R = neigh.predict(imgs_test_a)


# Exportar el modelo 

joblib.dump(neigh, 'TrainedModel.sav', compress=0,protocol=2) # Guardo el modelo.

print("Model Exported ! ")