# demographics-dnn

## usage

python3 demo.py --input selfie.mp4

Notes on argument parameters
Some labels, dimensions and channels must be specified depending on the following observations:

    - INPUT: path to input video
    - OUTPUT: path to output video, optional, e.g. --output output.avi
    - DISPLAY: whether to display detections while producing them, optional, e.g, default = 1 : show, 0 = do not show, --display 0
    - CONFIDENCE: minimum prediction probability of certainty to count as detection, optional, e.g. --confidence 0.9
    - CAMERA: whether video is being taken by frontal or rear smartphone camera, optional, e.g. --camera r
    - ORIENTATION: whether the counter line is horizontal or vertical, optional, e.g. --orientation v
    - MODEL TYPE... type a = XCEPTION and Balaji models;  any other letter refers to PRIYA and VGG Net models, optional, e.g. --model-type a
    - CHANNEL = XCEPTION & VGGNet MODELS ARE ON BGR(3), REGULAR MODEL IS GRAYSCALE(1) CHANNELS MUST BE CHANGED, optional, e.g. --channel 3
    - DIMENSION = all models have input size (48, 48) except for regular VGGNet (96, 96) and mini_Xception_v2 (62, 62), optional, e.g. --dimension 48



