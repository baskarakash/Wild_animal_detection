#!/usr/bin/env python
# coding: utf-8

# In[41]:


import cv2
import numpy as np
import os
import sys
import cv2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
from keras.utils import np_utils
from tensorflow.keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D
from tensorflow.keras.optimizers import Adam
import itertools
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from pathlib import Path
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from tensorflow.keras.optimizers import legacy 


# In[2]:


train_dir=r"C:\Users\Akash Baskar\Desktop\animal_detection"
val_dir=r"C:\Users\Akash Baskar\Desktop\animal_detection_train"


# In[17]:


class_labels=os.listdir(train_dir)
print(class_labels)
IMAGE_SIZE=30


# In[18]:


def create_training_data():
    training_date = []
    for categories in class_labels:
        path = os.path.join(DATA_DIR,categories)
        class_num = c.index(categories)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array,(IMAGE_SIZE,IMAGE_SIZE))
                training_date.append([new_array,class_num])
            except:
                pass
    return training_dat


# In[19]:


train_total=0
for label in class_labels:
    total=len(os.listdir(os.path.join(train_dir,label)))
    print(label,total)
    train_total+=total
print('Total....',train_total)


# In[20]:


val_total=0
for label in class_labels:
    total=len(os.listdir(os.path.join(val_dir,label)))
    print(label,total)
    val_total+=total
print('Total....',val_total)


# In[21]:


nb_train_samples=train_total
nb_val_samples=val_total
num_classes=5
img_rows=128
img_cols=128
channel=3


# In[22]:


x_train=[]
y_train=[]
i=0
j=0
for label in class_labels:
    image_names_train=os.listdir(os.path.join(train_dir,label))
    total=len(image_names_train)
    print(label,total)
    for image_name in image_names_train:
        try:
            img=image.load_img(os.path.join(train_dir,label,image_name),target_size=(img_rows,img_rows,channel))
            img=image.img_to_array(img)
            img=img/255
            x_train.append(img)
            y_train.append(j)
        except:
            pass
        i += 1
    j += 1
x_train=np.array(x_train)
y_train=np.array(y_train)
y_train=np_utils.to_categorical(y_train[:nb_train_samples],num_classes)
    


# In[23]:


x_test=[]
y_test=[]
i=0
j=0
for label in class_labels:
    image_names_train=os.listdir(os.path.join(val_dir,label))
    total=len(image_names_train)
    print(label,total)
    for image_name in image_names_train:
        try:
            img=image.load_img(os.path.join(val_dir,label,image_name),target_size=(img_rows,img_rows,channel))
            img=image.img_to_array(img)
            img=img/255
            x_test.append(img)
            y_test.append(j)
        except:
            pass
        i += 1
    j += 1
x_test=np.array(x_test)
y_test=np.array(y_test)
y_test=np_utils.to_categorical(y_test[:nb_val_samples],num_classes)
    


# In[24]:


print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)


# In[25]:


plt.imshow(x_test[10])


# In[26]:


model=Sequential()


# In[27]:


model.add(Conv2D(filters=32,kernel_size=(3,3),padding='valid',activation='relu',input_shape=(128,128,3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))
          
model.add(Conv2D(filters=64,kernel_size=(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))

model.add(Conv2D(filters=128,kernel_size=(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1)) 
          
model.add(Conv2D(filters=256,kernel_size=(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1)) 
          
model.add(Flatten())
model.add(Dense(1026,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(num_classes,activation='softmax'))


# In[28]:


import tensorflow as tf
model.compile(
    loss='categorical_crossentropy',
    optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.001,decay=1e-04),
    metrics=['accuracy']
)


# In[29]:


model.summary()


# In[30]:


model.fit(
    x_train,
    y_train,
    batch_size=10,
    epochs=5,
    validation_data=(x_test,y_test),
    shuffle=True
    
)
model.save('animal.model')


# In[31]:


def prepare(filepath):
    training_date = []
    
    img_array = cv2.imread(filepath,cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array,(IMAGE_SIZE,IMAGE_SIZE))
    new_image =  new_array.reshape(-1,IMAGE_SIZE,IMAGE_SIZE,1)
    return new_image


# In[32]:


model = tf.keras.models.load_model('animal.model')


# In[42]:


filepath = '‪‪C:/Users/Akash Baskar/Desktop/wildlife/leopard.jpg'
img_array = cv2.imread(filepath,cv2.IMREAD_GRAYSCALE)

plt.imshow(img_array)


# In[43]:


test = model.predict([prepare(filepath=filepath )])


# In[ ]:


y_pred=model.predict(x_test,batch_size=1)
print(y_pred)


# In[ ]:





# In[ ]:


y_predict=[]
for i in range(0,len(y_pred)):
    y_predict.append(int(np.argmax(y_pred[i])))
len(y_predict)


# In[ ]:


y_true=[]
for i in range(0,len(y_test)):
    y_true.append(int(np.argmax(y_test[i])))
len(y_true)


# In[ ]:


def plot_confusion_matrix(cm,classes,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    plt.figure(figsize=(15,7))
    plt.imshow(cm,interpolation='nearest',cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks=np.arange(len(classes))
    plt.xticks(tick_marks,classes,rotation=45)
    plt.yticks(tick_marks,classes)
    
    fmt='d'
    thresh=cm.max()/2
    for i,j in itertools.product(range(cm.shape[0]),range(cm.shape[1])):
        plt.text(j,i,format(cm[i,j],fmt),
                 horizontalalignment="center",
                 color="white" if cm[i,j]>thresh else"black")
    plt.ylabel('Actual label')
    plt.xlabel('predicted label')
    plt.tight_layout()
    plt.show()


# In[ ]:


cm_plot_labels=class_labels


# In[ ]:


print(classification_report(y_true=y_true,y_pred=y_predict))


# In[ ]:


cm=confusion_matrix(y_true=y_true,y_pred=y_predict)
plot_confusion_matrix(cm,cm_plot_labels,title='Confusion Matrix')


# In[ ]:


score=model.evaluate(x=x_test,y=y_test,batch_size=32)
print("Test Accuracy: ",score[1])


# In[ ]:


score=model.evaluate(x=x_train,y=y_train,batch_size=32)
print("Test Accuracy: ",score[1])


# In[ ]:




