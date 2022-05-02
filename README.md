# drowsiness-detection-system
A drowsiness detection system implemented on a Raspberry Pi 4 using Machine Learning in Python.

For the script to run for it's intended purpose, the following code segment needs to be implemented to run on startup of the Raspberry Pi:
```python
python drowsiness_detection_pi_hog.py --shape-predictor shape_predictor_68_face_landmarks.dat --alarm 1
```
The Raspberry Pi also needs to be manipulated into thinking that it is connected to an external monitor. This can be forced by changing the code in ```config.txt``` to ```hdmi_force_hotplug=1```.

The Raspberry Pi 4 can be used in a cars USB/Auxillary power outlet for power. The script containing the algorithm will then run continuously and monitor the users face and extract their eyes until the power is switched off. When a user closes their eyes for a period of approximatley 2.5 seconds, a loud audible alarm is sounded through a Piezo buzzer connected to the Raspberry Pi hardware pins.

Full details can be found in the Project Report.
