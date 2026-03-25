# crab_image_detection2026
Crab image detection for MATE ROV 2026. Uses Picam &amp; Raspberry Pi to stream video frames to laptop via ethernet, then frames are processed by YOLOv8 model.

best.pt - YOLOv8 image detection model trained on images of Rock Crabs. Green Crabs, and Jonah Crabs from the internet and the images from MATE.

pi_send.py - streams 320,240 video frames to laptop by encoding them as jpeg images and sending them via ethernet

py_stream.py - captures aformentioned frames and processes them using OpenCV and YOLOv8 model

requirements.txt - all packages needed in a venv for this to work
