import argparse
import time
import timeit
import datetime

import cv2
import dlib
import imutils
import numpy as np
from imutils import face_utils
from imutils.video import VideoStream
from gpiozero import Buzzer


# define the Euclidean distance with a function
def euclidean_dist(a, b):
    # compute and return the euclidean distance between point a and point b
    return np.linalg.norm(a - b)

# compute the Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye):
    # compute the euclidean distances between the two sets of vertical eye landmarks (x, y) coordinates
    A = euclidean_dist(eye[1], eye[5])
    B = euclidean_dist(eye[2], eye[4])

    # compute the euclidean distance between the horizontal eye landmark (x, y) coordinates
    C = euclidean_dist(eye[0], eye[3])

    # compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)

    # return the ratio
    return ear

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True, help="path to facial landmark predictor")
ap.add_argument("-a", "--alarm", type=int, default=0, help="boolean used to indicate if alarm should be used")
args = vars(ap.parse_args())

# set off buzzer when ear drops
if args["alarm"] > 0:
    bz = Buzzer(13, active_high=True, initial_value=False) # this is the GPIO number for the pi pin
    print("Using buzzer alarm...")
    

ear_threshold = 0.6
ear_frames = 11

counter = 0
alarm_on = False

print('Loading facial landmark predictor...')
start_time = timeit.default_timer()  # start timing model speed
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])
end_time = timeit.default_timer()  # end of execution time for model

total_fps = 0  # count total number of frames
frame_count = 0  # keep track of total frames per second
execution_time = end_time - start_time


(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

print("Starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(1.0)
print("Video stream started")

while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=450)
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    fps_start = time.time()  # start timer for fps before detector
    rects = detector(grey, 0)
    fps_end = time.time()  # end timer for fps after detector

    fps = 1 / (fps_end - fps_start)  # get current FPS
    total_fps += fps  # add fps to total fps
    frame_count += 1  # increment frame count
    
    
    for rect in rects:
        shape = predictor(grey, rect)
        shape = face_utils.shape_to_np(shape)
        
        left_eye = shape[lStart:lEnd]
        right_eye = shape[rStart:rEnd]
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)

        # average the ear together for both eyes
        ear = (left_ear + right_ear)

        # compute the convex hull for the left and right eye, then visualize each of the eyes
        #leftEyeHull = cv2.convexHull(left_eye)
        #rightEyeHull = cv2.convexHull(right_eye)
        #cv2.drawContours(frame, [leftEyeHull], -1, (255, 255, 0), 1)
        #cv2.drawContours(frame, [rightEyeHull], -1, (255, 255, 0), 1)
        
        if ear < ear_threshold:  # less than 0.5
            counter += 1  # count the frames

            # if the eyes were closed for sufficient number of frames, then sound the alarm
            if counter >= ear_frames: # check if the number of frames exceeds the threshold
                # if the alarm is not on, turn it on
                if not alarm_on:
                    alarm_on = True

                    # check to see if the buzzer should be sounded
                    if args["alarm"] > 0:
                        bz.beep(on_time=0.1, off_time=0.1, n=10)
                # cv2.putText(frame, "ALERT", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # else the ear ratio is not below the blink threshold so reset the counter and alarm
        else:
            counter = 0
            alarm_on = False

        #draw ear on the frame for debugging and setting the ear threshold and frame counters
        #cv2.putText(frame, "EAR: {:.3f}".format(ear), (350, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    # draw frames per second
    #cv2.putText(frame, "FPS: {:.0f}".format(fps), (350, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    #show the frame
    cv2.imshow('frame', frame)

    key = cv2.waitKey(1) & 0xFF

    # if q key is pressed, break the loop
    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()

program_end_time = timeit.default_timer() # end of program total run time
total_run_time = program_end_time - start_time # calculate total run time

avg_fps = total_fps / frame_count
print("Total run time: " + str(datetime.timedelta(seconds=round(total_run_time))))
print("Model execution time: {:.2f} seconds".format(execution_time))
print("Average FPS: " + str(int(avg_fps)))
