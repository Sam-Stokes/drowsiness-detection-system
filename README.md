# drowsiness-detection-system
A drowsiness detection system implemented on a Raspberry Pi 4 using Machine Learning in Python.

The code for running the script on start up is:

python drowsiness_detection_pi_hog.py --shape-predictor shape_predictor_68_face_landmarks.dat --alarm 1

and need to create a way of running on start up.

Also need to trick it into t hinking it is plugged into a monitor. hdmi_force_hotplug=1 in config.txt

The Raspberry Pi 4 can be used in a cars USB/Auxillary power outlet for power. The script containing the algorithm will then run continuously and monitor the users face and extract their eyes until the power is switched off. When a user closes their eyes for a period of approximatley 2.5 seconds, a loud audible alarm is sounded through a Piezo buzzer connected to the Raspberry Pi hardware pins.
