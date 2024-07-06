import cv2
import time
import datetime 
#security camera 
cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')
detection = False
recording = True
detection_stopped_time = None
timer_started = False
SECONDS_TO_RECORD_AFTER_DETECTION = 5

frame_size = (int(cap.get(3)),int(cap.get(4)))
fourcc = cv2.VideoWriter_fourcc(*"mpv4")

while True:
    
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,1.3,5)
    bodies = body_cascade.detectMultiScale(gray,1.3,5)
    
    if len(faces)+ len(bodies) > 0:
        if detection:
            timer_started = False
        else:
            detection = True
            current_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            out = cv2.VideoWriter(f"{current_time}.mp4",fourcc, 20, frame_size)
            print("started rec")
    elif detection:
        if timer_started:
            if time.time() - detection_stopped_time >= SECONDS_TO_RECORD_AFTER_DETECTION:
                detection = False 
                timer_started = False
                out.release()
                print("stopped rec")
        else:
            timer_started = True
            detection_stopped_time = time.time()
    if detection:
        out.write(frame)
    

    #cv2.imshow('Camera', frame) 
    #we dont need to show the frame because it is being recorded
    
    if cv2.waitKey(1) & 0xFF == ord('x'):
        break
out.releae()
#release the recording
cap.release()
#release the camera
cv2.destroyAllWindows()
#destroy all windows