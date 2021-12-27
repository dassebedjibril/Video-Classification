#!/usr/bin/env python
# coding: utf-8


from keras.models import load_model
from collections import deque
import numpy as np
import pickle
import cv2



model = load_model(r"C:\Users\DASSEBE\Sport video classifier\Project_1\VideoClassificationModel")
lb = pickle.loads(open(r"C:\Users\DASSEBE\Sport video classifier\Project_1\videoclassificationbinarizer.pickle", "rb").read())
outputvideo = r"C:\Users\DASSEBE\Sport video classifier\Project_1\demo_output.avi"
mean = np.array([123.68, 116.779, 103.99][::1], dtype = "float32")
Queue = deque(maxlen = 128)




capture_video = cv2.VideoCapture(r"C:\Users\DASSEBE\Sport video classifier\Project_1\video_de_test_1.mp4")
writer = None
(Width, Height) = (None, None)

while True:
    (taken, frame) = capture_video.read()
    if not taken:
        break
    if Width is None or Height is None:
        (Width, Height) = frame.shape[:2]
        
    output = frame.copy()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (224, 244)).astype("float32")
    frame = frame.reshape(1, 224, 244, 3)
    frame -= mean
    preds = model.predict(np.expand_dims(frame, axis= 0))[0]
    Queue.append(preds)
    results = np.array(Queue).mean(axis = 0)
    i = np.argmax(results)
    label = lb.classes_[i]
    text = "They are playing: {}".format(label)
    cv2.putText(output, text, (45, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (255, 0, 0), 5)
    
    if writer is None:
        fourcc = cv2.VideoWriter("MJPG")
        writer = cv2.VideoWriter("outputvideo", fourcc, 30, (Width, Height), True)
    writer.write(output)
    cv2.imshow("In progress", output)
    key = cv2.waitKey(1)  & 0xFF
    
    if key == ord("q"):
        break
        

print("Finalizing....")
writer.release()
capture_video.release()



from tensorflow.keras.models import load_model
from collections import deque
import numpy as np
import pickle
import cv2

model_path = r"C:\Users\DASSEBE\Sport video classifier\Project_1\VideoClassificationModel"
lb_path = r"C:\Users\DASSEBE\Sport video classifier\Project_1\videoclassificationbinarizer.pickle"
model = load_model(model_path)
lb = pickle.loads(open(lb_path, "rb").read())


mean = np.array([123.68, 116.779, 103.939][::1], dtype="float32")
Q = deque(maxlen=32)

video_path = r"C:\Users\DASSEBE\Sport video classifier\Project_1\video_de_test_1.mp4"
output_video_path = r"C:\Users\DASSEBE\Sport video classifier\Project_1\demo_output.avi"
vs = cv2.VideoCapture(video_path)
writer = None
(W, H) = (None, None)
print('Predicting ...')
while True:
	(grabbed, frame) = vs.read()
	if not grabbed:
		break
	if W is None or H is None:
		(H, W) = frame.shape[:2]
	output = frame.copy()
	# Preprocessing each frame in the video
	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	frame = cv2.resize(frame, (244, 224)).astype("float32")
	frame -= mean

	# Predicting label for each frame
	preds = model.predict(np.expand_dims(frame, axis=0))[0]
	# Taking rolling average of predictions to avoid flickering predictions
	Q.append(preds)
	results = np.array(Q).mean(axis=0)
	i = np.argmax(results)
	label = lb.classes_[i]

	# Pasting the label on the output frame
	text = "activity: {}".format(label)
	cv2.putText(output, text, (35, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (0, 255, 0), 5)
	if writer is None:
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter(output_video_path, fourcc, 30, (W, H), True)
		writer.write(output)
writer.release()
vs.release()



