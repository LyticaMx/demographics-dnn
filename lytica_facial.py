'''
Description: Script made for processing an image and detecting facial emotion

Note: Please use this script when working with HDF5 format models
'''

from keras.preprocessing.image import img_to_array
from keras.models import load_model
import imutils
import cv2
import numpy as np
import sys
import ntpath
import dlib
from imutils.video import VideoStream
import json
import os
import time

class ImageProcessing:
    def image_preprocessing(img_path):
        # reading the frame
        orig_frame = cv2.imread(img_path)
        frame = cv2.imread(img_path)
        (h1, w1) = orig_frame.shape[:2]
        # maintaining the aspect ratio), and then grab the image dimensions
        if h1 > w1:
            orig_frame = imutils.resize(orig_frame, height=600)
            (h, w) = orig_frame.shape[:2]
            frame = imutils.resize(frame, height=600)
            (h, w) = frame.shape[:2]
        else:
            orig_frame = imutils.resize(orig_frame, width=600)
            (h, w) = orig_frame.shape[:2]
            frame = imutils.resize(frame, width=600)
            (h, w) = frame.shape[:2]

        return h, w, frame, orig_frame

    def show_image(img_path, orig_frame):
        cv2.imshow('test_face', orig_frame)
        cv2.waitKey(0)
        cv2.imwrite('output/' + ntpath.basename(img_path), orig_frame)

        if (cv2.waitKey(2000) & 0xFF == ord('q')):
            sys.exit("Thanks")
        cv2.destroyAllWindows()

    def initialize_stream(input):
        # if a video path was not supplied, grab a reference to the webcam
        if not input:
            print("[INFO] starting video stream...")
            vs = VideoStream(src=0).start()
            fps = vs.get(cv2.CAP_PROP_FPS)
            time.sleep(2.0)
        # otherwise, grab a reference to the video file
        else:
            print("[INFO] opening video file...")
            vs = cv2.VideoCapture(input)
            fps = vs.get(cv2.CAP_PROP_FPS)

        # initialize the frame dimensions (we'll set them as soon as we read
        # the first frame from the video)
        W = None
        H = None

        # initialize the video writer (we'll instantiate later if need be)
        writer = None

        return vs, fps, W, H, writer

    def read_frame(vs):
        # grab the next frame and handle if we are reading from either
        # VideoCapture or VideoStream
        frame = vs.read()
        frame = frame[1] if input else frame

        return frame

    def processFrame(frame, W, H, output, writer, model_channel, camera):
        # resize the frame to have a maximum width of 500 pixels (the
        # less data we have, the faster we can process it), then convert
        # the frame from BGR to RGB for dlib
        (h1, w1) = frame.shape[:2]
        # maintaining the aspect ratio), and then grab the image dimensions
        if h1 > w1:
            frame = imutils.resize(frame, height=500)
        else:
            frame = imutils.resize(frame, width=500)

        if camera == 'f':
            frame = cv2.flip(frame, -1)

        if model_channel == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # may be RBG, GRAYSCALE OR WHATEVS TBH MAN
        else:
            gray = frame


        # if the frame dimensions are empty, set them
        if W is None or H is None:
            (H, W) = frame.shape[:2]

        # if we are supposed to be writing a video to disk, initialize
        # the writer
        if output is not None and writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(output, fourcc, 30, (W, H), True)

        return frame, gray, writer, H, W

    def end_stream(fps, writer, vs):
        # stop the timer and display FPS information
        print("[INFO] approx. FPS: {:.2f}".format(fps))

        # check to see if we need to release the video writer pointer
        if writer is not None:
            writer.release()

        vs.release()

        # close any open windows
        cv2.destroyAllWindows()


class Utils:
    def jsonify(json_emocion, json_demografico):
        print(json_demografico, json_emocion)
        json_facial = dict(json_demografico).update(json_emocion)
        print(json_facial)
        json_facial = json.dumps(json_facial)

        return json_facial

    def dictionize(info, contador_info, emotion_info, demographics_info):
        info.update(contador_info)
        info.update(emotion_info)
        info.update(demographics_info)

        return info

    def listify(contador_info, emotion_info, demographics_info, face_info):
        info = contador_info + face_info + demographics_info + emotion_info

        return info


class Demographics:
    def loadModels():
        # cargamos los detectores y redes
        directorio = os.path.dirname(os.path.abspath(__file__))+'/static/models'
        detector = dlib.get_frontal_face_detector()
        #predictor = dlib.shape_predictor((directorio + "/shape_predictor_68_face_landmarks.dat"))
        faceProto = directorio + "/opencv_face_detector.pbtxt"
        faceModel = directorio + "/opencv_face_detector_uint8.pb"
        ageProto = directorio + "/age_deploy.prototxt"
        ageModel = directorio + "/age_net.caffemodel"
        genderProto = directorio + "/gender_deploy.prototxt"
        genderModel = directorio + "/gender_net.caffemodel"

        return detector, faceProto, faceModel, ageProto, ageModel, genderProto, genderModel

    def modelHipothesis(ageModel, ageProto, genderModel, genderProto, faceModel, faceProto):
        MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
        ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
        genderList = ['Male', 'Female']

        # Cargamos las R-CNN para Edad y Sexo
        ageNet = cv2.dnn.readNet(ageModel, ageProto)
        genderNet = cv2.dnn.readNet(genderModel, genderProto)
        faceNet = cv2.dnn.readNet(faceModel, faceProto)
        padding = 20

        return MODEL_MEAN_VALUES, ageList, genderList, ageNet, genderNet, faceNet, padding

    def detectFaces(gray, detector):
        # Primero detecto la cara de las personas
        faces = detector(gray)

        return faces

    def haarDetection(frame, face_detection):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detection.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(40, 40), flags=cv2.CASCADE_SCALE_IMAGE)

        return faces

    def genderDetection(blob, genderNet, genderList):
        # primero convierto las "caras" en blobs y luego las paso por las redes neuronales de caracteristicas
        genderNet.setInput(blob)
        genderPreds = genderNet.forward()
        gender = genderList[genderPreds[0].argmax()]
        confianza = max(genderPreds[0])

        return gender, confianza

    def ageDetection(blob, ageNet, ageList):
        ageNet.setInput(blob)
        agePreds = ageNet.forward()
        age = ageList[agePreds[0].argmax()]
        confianza = max(agePreds[0])

        return age, confianza

    def facialLandmarks(frame, gray, face, predictor):
        # aqui dibujare los puntos de la cara de la persona
        landmarks = predictor(gray, face)

        # teniendo la cara pinto los 68 puntos de interes
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            cv2.circle(frame, (x, y), 1, (153, 255, 221), -1)

        return frame, landmarks

    def demographicDetection(frame, faces, MODEL_MEAN_VALUES, genderNet, ageNet, genderList, ageList, Haar):
        face_num = 1
        # info = {}
        if len(faces) > 0:
            for face in faces:
                if Haar:
                    (fX, fY, fW, fH) = face

                else:
                    x1 = face.left()
                    y1 = face.top()
                    x2 = face.right()
                    y2 = face.bottom()

                # basicamente saco los vertices del rectangulo que delimitara la cara, los uno y los pinto
                cv2.rectangle(frame, (fX, fY), (fX + fW, fY + fH), (255, 128, 255), 1)
                # estas coordenadas serán guardadas como objetos (caras) detectados
                roi = frame[fY:fY + fH, fX:fX + fW]

                blob = cv2.dnn.blobFromImage(roi, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

                gender, gender_confianza = Demographics.genderDetection(blob, genderNet, genderList)
                age, age_confianza = Demographics.ageDetection(blob, ageNet, ageList)

                # formato de salida para el texto de sexo y edad
                label = "{}: {}".format(gender, age)

                # formato de salida
                cv2.putText(frame, label, (fX, fY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 1,
                            cv2.LINE_AA)

                # info = {"Persona " + str(face_num): {"Genero": gender, "Rango de Edad": age}}
                info = [gender, age, min(gender_confianza, age_confianza)]

        else:
            info = ["", "", ""]

        return frame, info


class EmotionDetection:
    #FUNCION PARA CARGAR MODELOS DE DETECCION FACIAL Y DE EMOCIONES
    def emotion_model(detection_model_path, emotion_model_path, model_type):
        # hyper-parameters for bounding boxes shape
        # loading models
        face_detection = cv2.CascadeClassifier(detection_model_path)
        emotion_classifier = load_model(emotion_model_path, compile=False)

        if model_type == 'a':
            EMOTIONS = ["angry", "disgust", "scared", "happy", "sad", "surprised",
                        "neutral"]  # XCEPTION models & BALAJI  LABELS
        else:
            EMOTIONS = ["angry", "disgust", "scared", "happy", "neutral", "sad",
                        "surprised"]  # PRIYA and VGG Net models LABELS

        return face_detection, emotion_classifier, EMOTIONS

    def emotion_detect(face_detection, emotion_classifier, EMOTIONS, frame, orig_frame, model_channel, input_dim):
        faces = face_detection.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20),
                                                flags=cv2.CASCADE_SCALE_IMAGE)
        # RUN EMOTION DETECTOR IF FACES WHERE FOUND IN FRAME
        if len(faces) > 0:
            # INITIALIZE A COUNTER IF THERE ARE MORE THAN 1 PEOPLE IN THE FRAME
            people_counter = 0
            for face in faces:
                # face is acquired from picture and analyzed alone (ROI stands for region of interest)
                (fX, fY, fW, fH) = face
                roi = frame[fY:fY + fH, fX:fX + fW]
                roi = cv2.resize(roi, (input_dim, input_dim))
                if model_channel == 1:
                    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                roi = roi.astype("float") / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)
                preds = emotion_classifier.predict(roi)[0]
                emotion_probability = np.max(preds)
                label = EMOTIONS[preds.argmax()]
                cv2.putText(orig_frame, label, (fX, fY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.rectangle(orig_frame, (fX, fY), (fX + fW, fY + fH), (0, 0, 255), 2)
                people_counter += 1
                print("Person {} is {} ({:.2f}% confidence).".format(people_counter, label, emotion_probability * 100))

    def video_emotion_detect(frame, gray, faces, emotion_classifier, EMOTIONS, model_channel, input_dim, W, H, writer):
        frameClone = frame.copy()
        #faces = face_detection.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(20, 20),
                                                # maxSize=(80, 80), flags=cv2.CASCADE_SCALE_IMAGE)
        canvas = np.zeros((250, 300, 3), dtype="uint8")
        # info = {}

        if len(faces) > 0:
            # INITIALIZE A COUNTER IF THERE ARE MORE THAN 1 PEOPLE IN THE FRAME
            people_counter = 0
            for face in faces:
                # face is acquired from picture and analyzed alone (ROI stands for region of interest)
                (fX, fY, fW, fH) = face
                roi = frame[fY:fY + fH, fX:fX + fW]
                roi = cv2.resize(roi, (input_dim, input_dim))
                if model_channel == 1:
                    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                roi = roi.astype("float") / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)
                preds = emotion_classifier.predict(roi)[0]
                emotion_probability = np.max(preds)
                label = EMOTIONS[preds.argmax()]
                cv2.putText(frameClone, label, (fX, fY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
                cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH), (0, 0, 255), 2)
                people_counter += 1
                # print("Person {} is {} ({:.2f}% confidence).".format(people_counter, label, emotion_probability * 100))
                # info = {"Person "+str(people_counter): {"emocion": label, "confianza": emotion_probability*100}}
                info = [label, emotion_probability]

            for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, preds)):
                # construct the label text
                text = "{}: {:.2f}%".format(emotion, prob * 100)
                text2 = "{}: {:.2f}%".format(label, emotion_probability * 100)
                w = int(prob * 100)
                cv2.rectangle(frameClone, (7, (i * 35) + 5),
                              (w, (i * 35) + 35), (0, 0, 255), -1)
                cv2.putText(frameClone, text, (10, (i * 35) + 23),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            (0, 255, 255), 1)
                cv2.putText(frameClone, text2, (fX, fY - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
                cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH),
                              (0, 0, 255), 2)


        else:
            #info = {'Detecciones': 0}
            info = ["", ""]

        # if we were to return json
        # json_info = json.dumps(info)

        return writer, frameClone, info

