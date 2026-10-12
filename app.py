import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="AI Facial Emotion Recognition",
    page_icon="😊",
    layout="wide"
)

# --------------------------------------------------
# Emotion labels
# --------------------------------------------------

EMOTIONS = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Sad",
    "Surprise",
    "Neutral"
]

# --------------------------------------------------
# Load model
# --------------------------------------------------

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("Model/emotion_detector.keras")


model = load_model()

# --------------------------------------------------
# Load OpenCV face detector
# --------------------------------------------------

@st.cache_resource
def load_face_detector():

    detector = cv2.CascadeClassifier(
        "haarcascade/haarcascade_frontalface_default.xml"
    )

    return detector


face_detector = load_face_detector()

# --------------------------------------------------
# Prediction function
# --------------------------------------------------

def predict_emotion(face):

    # Convert to grayscale
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

    # Resize
    gray = cv2.resize(gray, (48, 48))

    # Normalize
    gray = gray.astype("float32") / 255.0

    # Add dimensions
    gray = np.expand_dims(gray, axis=-1)
    gray = np.expand_dims(gray, axis=0)

    # Prediction
    predictions = model.predict(gray, verbose=0)[0]

    emotion_index = np.argmax(predictions)

    emotion = EMOTIONS[emotion_index]

    confidence = predictions[emotion_index]

    return emotion, confidence, predictions


# --------------------------------------------------
# Process image
# --------------------------------------------------

def process_image(image):

    image_array = np.array(image)

    # Convert RGB → BGR
    frame = cv2.cvtColor(
        image_array,
        cv2.COLOR_RGB2BGR
    )

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(40, 40)
    )

    results = []

    for (x, y, w, h) in faces:

        face = frame[y:y+h, x:x+w]

        emotion, confidence, predictions = predict_emotion(face)

        results.append({
            "box": (x, y, w, h),
            "emotion": emotion,
            "confidence": confidence,
            "predictions": predictions
        })

        # Draw bounding box
        cv2.rectangle(
            frame,
            (x, y),
            (x+w, y+h),
            (0, 255, 0),
            2
        )

        # Draw emotion
        label = f"{emotion}: {confidence * 100:.1f}%"

        cv2.putText(
            frame,
            label,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    output = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    return output, results


# --------------------------------------------------
# Sidebar
# --------------------------------------------------

st.sidebar.title("AI Facial Emotion Recognition")

st.sidebar.markdown(
    """
    ### About

    This application uses:

    - Convolutional Neural Network
    - TensorFlow/Keras
    - OpenCV
    - Streamlit

    The model predicts seven facial expressions.
    """
)

st.sidebar.markdown("---")

st.sidebar.write(
    "⚠️ Emotion predictions are model estimates "
    "and should not be treated as definitive measurements "
    "of a person's internal emotional state."
)

# --------------------------------------------------
# Main application
# --------------------------------------------------

st.title("😊 AI Facial Emotion Recognition")

st.write(
    "Upload a facial image and the AI model will "
    "detect faces and estimate the visible facial "
    "expression."
)

# --------------------------------------------------
# Upload image
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png", "jfif", "webp"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Original Image")

        st.image(
            image,
            use_container_width=True
        )

    # Process
    output_image, results = process_image(image)

    with col2:

        st.subheader("AI Detection")

        st.image(
            output_image,
            use_container_width=True
        )


# --------------------------------------------------
# Webcam capture
# --------------------------------------------------
st.subheader("Webcam")

camera_image = st.camera_input(
    "Take a picture"
)

if camera_image is not None:

    image = Image.open(camera_image).convert("RGB")

    output_image, results = process_image(image)

    st.image(
        output_image,
        caption="Emotion Detection",
        width="stretch"
    )

    if results:

        for result in results:

            st.success(
                f"Emotion: {result['emotion']}"
            )

            st.write(
                f"Confidence: "
                f"{result['confidence'] * 100:.2f}%"
            )
    # --------------------------------------------------
    # Results
    # --------------------------------------------------

    if results:

        st.subheader("Emotion Analysis")

        for i, result in enumerate(results):

            st.markdown(
                f"### Face {i + 1}: {result['emotion']}"
            )

            confidence = result["confidence"] * 100

            st.progress(
                float(result["confidence"])
            )

            st.write(
                f"Confidence: **{confidence:.2f}%**"
            )

            st.write("Emotion probabilities:")

            probabilities = result["predictions"]

            for emotion, probability in zip(
                EMOTIONS,
                probabilities
            ):

                st.write(
                    f"{emotion}: "
                    f"{probability * 100:.2f}%"
                )

    else:

        st.warning(
            "No face detected. Please upload "
            "a clearer facial image."
        )

else:

    st.info(
        "Upload an image to begin emotion recognition."
    )