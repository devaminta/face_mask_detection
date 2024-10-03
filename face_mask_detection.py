# from zipfile import ZipFile
# dataset="full_data.zip"
# with ZipFile(dataset,"r") as zip:
#     zip.extractall()
#     print("The dataset is extracted")

# importing the dependencies

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mping
import cv2
from PIL import Image
from sklearn.model_selection import train_test_split


with_mask_files=os.listdir("data/with_mask")
with_out_mask_files=os.listdir("data/without_mask")

 
print("the length of with mask file",len(with_mask_files))  #3725
print("the length of with out mask file",len(with_out_mask_files)) #3828


# creating label fo the class of images
# with mask---> 1 , with out mask ----> 0
with_mask_labels=[1]* 3725
with_out_mask_labels=[0] * 3828

labels=with_mask_labels + with_out_mask_labels



# Displaying the images

# displaying with mask

# img=mping.imread("data/with_mask/with_mask_1.jpg")
# imgplot=plt.imshow(img)
# plt.show()



# displaying with out mask

# img=mping.imread("data/without_mask/without_mask_1.jpg")
# imgplot=plt.imshow(img)
# plt.show()

# image processing
# resize the images  and convert the images to numpy arrays
with_mask_path="data/with_mask/"
data =[]
for img_file in with_mask_files:
    image=Image.open(with_mask_path + img_file)
    image=image.resize((128,128))
    image=image.convert("RGB")
    image=np.array(image)
    data.append(image)
      
           
without_mask_path="data/without_mask/"
for img_file in with_out_mask_files:
    image=Image.open(without_mask_path + img_file)
    image=image.resize((128,128))
    image=image.convert("RGB")
    image=np.array(image)
    data.append(image)

# print(data[0])


# converting the image list and label to numpy arrays
X=np.array(data)
y=np.array(labels)


# print(X.shape)
# print(y.shape)  


# Train Test Split

X_train,X_test,Y_train,Y_test=train_test_split(X,y,test_size=0.2, random_state=2,stratify=y)


# scaling the data
X_train_scaled=X_train/255
X_test_scaled=X_test/255


# print(X_train_scaled[0])

# Building the convolution neural networks (CNN)

import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense,Dropout # type: ignore



num_of_classes=2

model=Sequential()


# Step 1 - Convolution
model.add(Conv2D(64, (3, 3), input_shape=(128, 128, 3), activation='relu'))


# Step 2 - Pooling
model.add(MaxPooling2D(pool_size=(2, 2)))

# # Adding a second convolutional layer
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# # Step 3 - Flattening
model.add(Flatten())

# # Step 4 - Full connection
model.add(Dense(units=128, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(units=64, activation='relu'))
model.add(Dropout(0.5))


model.add(Dense(num_of_classes, activation='sigmoid'))


# Compiling the CNN
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])



# training the neural networks 

history=model.fit(X_train_scaled,Y_train,validation_split=0.1,epochs=5)

loss,accuracy=model.evaluate(X_test_scaled,Y_test)

print("Test acuracy",accuracy)


h=history

# plot the loss value


plt.plot(h.history["loss"], label="Train Loss")
plt.plot(h.history["val_loss"], label="validation Loss")
plt.legend()
plt.show()


# plot the accuracy value
plt.plot(h.history["acc"], label="Train Accuracy")
plt.plot(h.history["val_acc"], label="validation Accuracy")
plt.legend()
plt.show()

# predict 
input_image_Path=input("Path of the image to be predicted : ")
input_image=cv2.imread(input_image_Path)
cv2.imshow(input_image)

input_image_resized=cv2.resizee(input_image,(128,128))


input_image_scaled=input_image_resized/255

input_image_reshaped=np.reshape(input_image_scaled,[1,128,128,3])
input_prediction=model.predict(input_image_reshaped)

print(input_prediction)
input_pred_label=np.argmax(input_prediction)
if input_pred_label==1:
    print("the person in the image is wearing a mask")
else:
    print("The person in the image is not wearing a mask")