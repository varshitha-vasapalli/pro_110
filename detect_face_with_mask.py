# import the opencv library
import cv2
import numpy as np

import tensorflow as tf

model = tf.keras.models.load_model('C:/Users/SREE/Downloads/PRO-C110-Student-Boilerplate-main/PRO-C110-Student-Boilerplate-main/keras_model.h5')


# define a video capture object
vid = cv2.VideoCapture(0)
  
while(True):
      
    # Capture the video frame by frame
    ret, frame = vid.read()
    img = cv2.resize(frame,(224,224))
    test_image = np.arry(img,dtype = np.float32)
    test_image = np.expand_dims(test_image)
    normalised_image = test_image/255.0
    prediction = model.predict(normalised_image)
    predict_class=np.argmx(prediction,axis=1)
    print("prediction:",predict_class)
    # Display the resulting frame
    cv2.imshow('frame', frame)
      
    # Quit window with spacebar
    key = cv2.waitKey(1)
    
    if key == 32:
        break
  
# After the loop release the cap object
vid.release()

# Destroy all the windows
cv2.destroyAllWindows()