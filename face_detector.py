import numpy as np
import cv2
from flask import Flask, render_template, Response
import tensorflow as tf
from keras.models import load_model
from tensorflow.python.keras.saving import hdf5_format
from tensorflow.python.keras.saving.saved_model import load as saved_model_load


# Eyes recognition
app = Flask(__name__)
model = tf.keras.models.load_model('./Models/CNN_model.h5', 
                                   custom_objects={'Functional':tf.keras.models.Model})

cap = cv2.VideoCapture(0)

def generate():
    while True:
        success, frame = cap.read()
        if not success:
            break
        else: 
            faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
            faces = faceCascade.detectMultiScale(frame, 1.1,7)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            for (x,y,w,h) in faces: 
                cv2.rectangle(frame, (x,y), (x+w, y+h), (255, 0, 0), 2)
                roi_gray= gray[y:y+h, x:x+w]
                roi_color= frame[y:y+h, x:x+w]
                eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3)
                for (ex, ey, ew,eh) in eyes:
                    eyes_roi = roi_color[ey:ey+eh, ex:ex+ew]
                    cv2.rectangle(roi_color, (ex,ey), (ex+ew, ey+eh), (0, 255, 0), 2)
            
            final_image = cv2.resize(eyes_roi, (180,180))
            final_image = np.expand_dims(final_image, axis = 0)
            
            Predictions = model.predict(final_image)
            if (round(Predictions[0,1])>0):
                status = "Ojos abiertos: " + str(Predictions[0,1])
                x1, y1, w1, h1 = 0,0,175,175
                cv2.putText(frame, status, (x1+int(w1/2), y1+int(h1/2)), cv2.FONT_HERSHEY_PLAIN, 4, (0,255,0),2)
        
            else:
                status = "Ojos cerrados: " + str(Predictions[0,1])
                x1, y1, w1, h1 = 0,0,175,175
                cv2.putText(frame, status, (x1+int(w1/2), y1+int(h1/2)), cv2.FONT_HERSHEY_PLAIN, 4, (255,0,255),2)
        
            ret, buffer=cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield(b'--frame\r\n' 
            b'Content-Type: image/jpeg\r\n\r\n'+ frame +b'\r\n')


# App creation
@app.route('/video')
def video():
    return Response(generate(), mimetype = 'multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return render_template('Index.html')

if __name__=="__main__":
    app.run(debug=True)

cap.release()