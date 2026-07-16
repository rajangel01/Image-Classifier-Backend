import numpy as np
from tensorflow.keras.models import load_model
from utils import preprocess_image

class_names = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

model = load_model("models/model.keras")

def predict_image(path):
    img = preprocess_image(path)

    prediction = model.predict(img)

    predicted_index = np.argmax(prediction[0])
    confidence = float(np.max(prediction[0]) * 100)

    return {
        "class": class_names[predicted_index],
        "confidence": round(confidence, 2)
    }