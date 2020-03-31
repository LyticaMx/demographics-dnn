'''
 Description: People counter and facial recognition tool
Terminal usage...

a)To read and write back out to video:
 python demo.py --input videos/example_01.mp4 --output output/output_01.avi

 b)To read from webcam and write back out to disk:
 python demo.py --output output/webcam_output.avi

 Notes on optional parameters
  Some labels, dimensions and channels must be specified depending on the following observations:

    - INPUT: path to input video
    - OUTPUT: path to output video
    - CONFIDENCE: minimum prediction probability of certainty to count as detection
    - CAMERA: whether video is being taken by frontal or rear smartphone camera
    - ORIENTATION: whether the counter line is horizontal or vertical
    - MODEL TYPE... type a = XCEPTION and Balaji models;  any other letter refers to PRIYA and VGG Net models
    - CHANNEL = XCEPTION & VGGNet MODELS ARE ON BGR(3), REGULAR MODEL IS GRAYSCALE(1) CHANNELS MUST BE CHANGED
    - DIMENSION = all models have input size (48, 48) except for regular VGGNet (96, 96) and mini_Xception_v2 (62, 62)
    - DISPLAY: whether to display detections while producing them, default = 1 : show, 0 = do not show

'''

# import the necessary packages
import argparse
import cv2
import datetime
import imutils
from imutils.video import FileVideoStream
import time
from lytica_facial import EmotionDetection, ImageProcessing, Demographics

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str, help="path to optional input video file")
ap.add_argument("-o", "--output", type=str, help="path to optional output video file")
ap.add_argument("--confidence", type=float, default=0.71, help="minimum probability to filter weak detections")
ap.add_argument("-cam", "--camera", type=str, default='r', help="specify which phone camera did you use... frontal (f) or rear (r)")
ap.add_argument("--orientation", type=str, default='v',help="counter line orientation, vertical or horizontal")
ap.add_argument("-t", "--type", type=str, default='a', help="specify model type")
ap.add_argument("--channel", type=int, default=1, help="channel of images, for BGR type 3")
ap.add_argument("-d", "--dimension", type=int, default=48, help="dimension of neural net input square image")
ap.add_argument('-ds', '--display', type=int, default=1, help='Display the detected images using OpenCV. This reduces FPS')
args = vars(ap.parse_args())


# LOADING MODELS AND PARAMETERS
# GENERAL PARAMETERS
output = args["output"]
model_channel = args["channel"]
# for emotion detection
detection_model_path = 'static/models/haarcascade_files/haarcascade_frontalface_default.xml'
model_type = args["type"]
input_dim = args["dimension"]

fvs = FileVideoStream(args["input"]).start()
time.sleep(3.0)

# CREATING MODELS
# for facial detection
face_detection = cv2.CascadeClassifier(detection_model_path)

# for facial demographics
detector, faceProto, faceModel, ageProto, ageModel, genderProto, genderModel = Demographics.loadModels()
# we load model constants
MODEL_MEAN_VALUES, ageList, genderList, ageNet, genderNet, faceNet, padding = Demographics.modelHipothesis(ageModel,
                                                                                                           ageProto,
                                                                                                           genderModel,
                                                                                                           genderProto,
                                                                                                           faceModel,
                                                                                                           faceProto)

frame = fvs.read()
(H, W) = frame.shape[:2]

if args["output"] is not None:
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(args["output"], fourcc, 30, (W, H), True)

else:
    writer = None

start_time = datetime.datetime.now()
num_frames = 0

# loop over frames from the video stream
while fvs.more():
    frame = fvs.read()

    # if we are viewing a video and we did not grab a frame then we
    # have reached the end of the video
    if args["input"] is not None and (fvs.Q.qsize() == 0):
        break

    frame = imutils.resize(frame, width=480)

    # FACIAL ATTRIBUTES
    # FIRST RUN FACIAL RECOGNITION
    # EMOTION DETECTION !!!
    faces = Demographics.haarDetection(frame, face_detection)

    # BINARY GENDER AND AGE PREDICTION !!!
    frame, demographics_info = Demographics.demographicDetection(frame, faces, MODEL_MEAN_VALUES,
                                                                 genderNet, ageNet, genderList, ageList, True)

    # Calculate Frames per second (FPS)
    num_frames += 1
    elapsed_time = (datetime.datetime.now() - start_time).total_seconds()
    fps = num_frames / elapsed_time

    if args["display"] > 0:
        # Display FPS on frame
        cv2.imshow('Multi-Threaded Detection', frame)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        print(demographics_info)
        print("frames processed: ", num_frames, "elapsed time: ",
              elapsed_time, "fps: ", str(int(fps)))

    # check to see if we should write the frame to disk
    if writer is not None:
        writer.write(frame)

# stop the timer and display FPS information
# ImageProcessing.end_stream(fps, writer, vs)
# cleanup
elapsed_time = (datetime.datetime.now() - start_time).total_seconds()
fps = num_frames / elapsed_time
print("mean fps: {}, elapsed time: {}".format(fps, elapsed_time))
cv2.destroyAllWindows()
fvs.stop()
