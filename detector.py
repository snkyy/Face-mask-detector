import cv2
import numpy as np
import tensorflow as tf

path1 = "deploy.prototxt"
path2 = "res10_300x300_ssd_iter_140000.caffemodel"
network = cv2.dnn.readNet(path1, path2)
mask_network = tf.keras.models.load_model("mask_detection_model")

capture = cv2.VideoCapture(0)
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# output = cv2.VideoWriter('output4.mp4', fourcc, 20.0, (1920, 1080))
if not capture.isOpened():
    print("Cannot open camera")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = capture.read()

    # ret = True if frame is read correctly
    if not ret:
        print("Can't receive frame (stream end?)")
        break

    # Constructing input blob
    (height, width) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224), (104.0, 177.0, 123.0))
    network.setInput(blob)
    detections = network.forward()

    faces = []
    faces_locations = []
    predictions = []

    # Loop over the detections
    for i in range(detections.shape[2]):
        # Extract the probability of the detection
        probability = detections[0, 0, i, 2]

        if probability > 0.5:
            # Computing coordinates of corners of the image
            outline = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
            (begX, begY, endX, endY) = outline.astype("int")

            # Ensure that the dimensions fit in the frame
            begX = max(0, begX)
            begY = max(0, begY)
            endX = min(width - 1, endX)
            endY = min(height - 1, endY)

            # Extract the face and modify it to fit in mask detection model
            face = frame[begY:endY, begX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = tf.keras.preprocessing.image.img_to_array(face)
            face = tf.keras.applications.mobilenet_v2.preprocess_input(face)

            # Store faces and their locations
            faces.append(face)
            faces_locations.append((begX, begY, endX, endY))

    # If any faces were detected make a prediction
    if len(faces) > 0:
        faces = np.array(faces, dtype="float32")
        predictions = mask_network.predict(faces, batch_size=32)

    # Display the frame accordingly to predictions
    for (location, prediction) in zip(faces_locations, predictions):

        # Set rectangle color to green if face is with a mask, red otherwise
        if prediction[0] > prediction[1]:
            color = (0, 255, 0)
        else:
            color = (0, 0, 255)

        cv2.rectangle(frame, (location[0], location[1]), (location[2], location[3]), color, 2)

    # Display the frame
    # output.write(frame)
    cv2.imshow('MaskDetector', frame)

    # Break the loop if 's' key is pressed
    key = cv2.waitKey(1)
    if key == ord("s"):
        break

capture.release()
# output.release()
cv2.destroyAllWindows()
