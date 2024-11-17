import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
import time
from tensorflow.keras.models import load_model

# Load the pre-trained model
facetracker = load_model('facetracker.h5')

# Streamlit Page Config
st.set_page_config(page_title="Face Tracker", page_icon=":guardsman:", layout="wide")

# Title
st.title("Real-Time Face Tracker")

# Set up webcam display for Streamlit
frame_placeholder = st.empty()  # Placeholder to update the image

# Function to preprocess the image
def preprocess_image(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized = tf.image.resize(image_rgb, (120, 120))
    return resized / 255.0

# Function to draw bounding boxes on images
def draw_bounding_box(frame, yhat):
    sample_coords = yhat[1][0]
    if yhat[0] > 0.5: 
        # Controls the main rectangle
        cv2.rectangle(frame, 
                      tuple(np.multiply(sample_coords[:2], [frame.shape[1], frame.shape[0]]).astype(int)),
                      tuple(np.multiply(sample_coords[2:], [frame.shape[1], frame.shape[0]]).astype(int)), 
                            (255, 0, 0), 2)
        # Controls the label rectangle
        cv2.rectangle(frame, 
                      tuple(np.add(np.multiply(sample_coords[:2], [frame.shape[1], frame.shape[0]]).astype(int), 
                                    [0, -30])),
                      tuple(np.add(np.multiply(sample_coords[:2], [frame.shape[1], frame.shape[0]]).astype(int),
                                    [80, 0])), 
                            (255, 0, 0), -1)

        # Controls the text rendered
        cv2.putText(frame, 'face', tuple(np.add(np.multiply(sample_coords[:2], [frame.shape[1], frame.shape[0]]).astype(int),
                                               [0, -5])),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    return frame

# Start video feed
cap = cv2.VideoCapture(0)

# Streamlit live video loop
while True:
    ret, frame = cap.read()
    if not ret:
        st.error("Unable to access webcam.")
        break

    # Preprocess the frame
    frame_resized = preprocess_image(frame)

    # Make predictions
    yhat = facetracker.predict(np.expand_dims(frame_resized, axis=0))

    # Draw bounding box on the frame
    frame_with_bbox = draw_bounding_box(frame, yhat)

    # Display the frame in Streamlit
    frame_placeholder.image(frame_with_bbox, channels="BGR", use_column_width=True)

    # Add a delay for smooth rendering (adjust if necessary)
    time.sleep(0.1)

    # Stop the loop if the user presses 'q' in the window
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
