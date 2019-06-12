import os,cv2

import numpy as np

from sklearn.utils import shuffle

from sklearn.model_selection import train_test_split



from keras import backend as K

K.set_image_dim_ordering('tf')



from keras.utils import np_utils

from keras.models import Sequential

from keras.layers.core import Dense, Dropout, Activation, Flatten

from keras.layers.convolutional import Convolution2D, MaxPooling2D

from keras.optimizers import SGD,RMSprop,adam



#PATH = os.getcwd()

# Define data path

data_dir_list = os.listdir('F:\\Smart Mirror\\Data directory\\')



img_rows=130

img_cols=150

num_channel=3

num_epoch=10



# Define the number of classes

num_classes = 2



img_data_list=[]



for dataset in data_dir_list:

    img_list=os.listdir('F:\\Smart Mirror\\Data directory\\'+ dataset)

    print ('Loaded the images of dataset-'+'{}\n'.format(dataset))

    for img in img_list:

        input_img=cv2.imread('F:\\Smart Mirror\\Data directory\\' + dataset + '\\'+ img )

        img_data_list.append(input_img)



img_data = np.array(img_data_list)

img_data = img_data.astype('float32')

img_data /= 255

print (img_data.shape)



# Define the number of classes

num_classes = 2



num_of_samples = img_data.shape[0]

labels = np.ones((num_of_samples,),dtype='int64')


# Assigning Labels

labels[0:52]=0

labels[52:103]=1


names = ['Lip Makeup','No Makeup']



# Convert class labels to One-hot encoding

Y = np_utils.to_categorical(labels, num_classes)



# Shuffle the dataset

x,y = shuffle(img_data,Y, random_state=2)

# Split the dataset

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)




# Defining the model
input_shape=img_data[0].shape


model = Sequential()



model.add(Convolution2D(32, 3,3,border_mode='same',input_shape=input_shape))

model.add(Activation('relu'))

model.add(Convolution2D(32, 3, 3))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.5))



model.add(Convolution2D(64, 3, 3))

model.add(Activation('relu'))

#model.add(Convolution2D(64, 3, 3))

#model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.5))



model.add(Flatten())

model.add(Dense(64))

model.add(Activation('relu'))

model.add(Dropout(0.5))

model.add(Dense(num_classes))

model.add(Activation('softmax'))



# sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

# model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=["accuracy"])

# model.compile(loss='categorical_crossentropy', optimizer='adadelta',metrics=["accuracy"])
model.compile(loss='categorical_crossentropy', optimizer='rmsprop',metrics=["accuracy"])



# Viewing model_configuration



model.summary()

model.get_config()

model.layers[0].get_config()

model.layers[0].input_shape			

model.layers[0].output_shape			

model.layers[0].get_weights()

np.shape(model.layers[0].get_weights()[0])

model.layers[0].trainable




# Training

hist = model.fit(X_train, y_train, batch_size=16, epochs=num_epoch, verbose=1, validation_data=(X_test, y_test))

#hist = model.fit(X_train, y_train, batch_size=32, nb_epoch=20,verbose=1, validation_split=0.2)


# Evaluating the model

score = model.evaluate(X_test, y_test, verbose=0)#, show_accuracy=True

print('Test Loss:', score[0])

print('Test accuracy:', score[1])



# Testing with an image in testing set

test_image = X_test[2:3]

print (test_image.shape)

print(model.predict(test_image))

print('Predicted:',names[model.predict_classes(test_image)[0]])

res= y_test[2:3]
#print(res)
for lis in res:
    for index,item in enumerate(lis):
        if item==1:
            ind = index
print('Original:',names[ind])



# Testing with a new image
'''

test_image = cv2.imread(r'F:\Smart mirror\Lipmakeup test\test7.jpg')

# test_image=cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)

cv2.imshow('image', test_image)

face_cascade = cv2.CascadeClassifier(r"F:\Anaconda\Library\etc\haarcascades\haarcascade_frontalface_default.xml")

gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)

face = face_cascade.detectMultiScale(gray, 1.2, 5)

for (x, y, w, h) in face:
    test_image = test_image[y:y + h, x:x + w]

test_image = cv2.resize(test_image, (130, 150))

test_image = np.array(test_image)

test_image = test_image.astype('float32')

test_image /= 255

print(test_image.shape)

test_image = np.expand_dims(test_image, axis=0)

print(test_image.shape)

# Predicting the test image

print((model.predict(test_image)))

print(model.predict_classes(test_image))

print('Predicted:', names[model.predict_classes(test_image)[0]])

cv2.waitKey(0)

cv2.destroyAllWindows()

'''