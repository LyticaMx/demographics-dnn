'''
Descritpion: Script para detectar caras en imagen (video) y predecir edad y sexo...

Terminal usage:
python edad_sexo.py --input path/to/input_video.avi --output path/to/output_video.avi


'''

import cv2
import argparse
from lytica_facial import Demographics, ImageProcessing

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str, help="path to optional input video file")
ap.add_argument("-o", "--output", type=str, help="path to optional output video file")
ap.add_argument("-cam", "--camera", type=str, default='r',
                help="specify which phone camera did you use... frontal (f) or rear (r)")
ap.add_argument("-c", "--channel", type=int, default=1, help="channel of images, for BGR type 3")
args = vars(ap.parse_args())

# if a video path was not supplied, grab a reference to the webcam, para seleccionar alguna otra camara como input
# revisar: https://answers.opencv.org/question/190710/how-do-i-choose-which-camera-i-am-accessing/
#initialize video stream
vs, fps, W, H, writer = ImageProcessing.initialize_stream(args["input"])

detector, predictor, faceProto, faceModel, ageProto, ageModel, genderProto, genderModel = Demographics.loadModels()
# we load model constants
MODEL_MEAN_VALUES, ageList, genderList, ageNet, genderNet, faceNet, padding = Demographics.modelHipothesis(ageModel,
                                                                                                           ageProto,
                                                                                                           genderModel,
                                                                                                           genderProto,
                                                                                                           faceModel,
                                                                                                           faceProto)

# loop over frames from the video file stream
while True:
    # grab the next frame and handle if we are reading from either
    # VideoCapture or VideoStream
    frame = ImageProcessing.read_frame(vs)

    # if we are viewing a video and we did not grab a frame then we
    # have reached the end of the video
    if args["input"] is not None and frame is None:
        break

    frame, gray, writer = ImageProcessing.processFrame(frame, W, H, args["output"], writer, args["channel"],
                                                       args["camera"])


    # FUNCION DETECCION CARAS
    faces = Demographics.detectFaces(gray, detector)
    print(faces)
    if faces:
        # FUNCION DETECCION GENERO BINARIO Y EDAD
        frame, json_label = Demographics.demographicDetection(frame, gray, faces, padding, MODEL_MEAN_VALUES,
                                                              genderNet, ageNet, genderList, ageList, predictor)

    # check to see if we should write the frame to disk
    if writer is not None:
        writer.write(frame)

    # muestro el video en vivo
    cv2.imshow("Demo", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# stop the timer and display FPS information
# fps.stop()
# print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
# print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# check to see if we need to release the video writer pointer
if writer is not None:
    writer.release()


# close any open windows
cv2.destroyAllWindows()
