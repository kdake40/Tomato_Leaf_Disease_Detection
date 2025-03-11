import os
import json
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from tensorflow.keras.preprocessing import image

# Initialize Flask app
app = Flask(__name__)

# Define the upload folder and allowed file extensions
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load the trained model
model = tf.keras.models.load_model('tomato.h5')

# Define prevention advice
PREVENTION = {
    "Early_Blight": "✅ Crop Rotation – Avoid planting tomatoes in the same spot every year.\n✅ Resistant Varieties – Grow resistant varieties like ‘Iron Lady’ or ‘Defiant’.\n✅ Proper Spacing – Ensure good airflow by planting tomatoes with enough space between them.\n✅ Mulching – Use organic mulch to prevent soil-borne spores from splashing onto leaves.\n✅ Avoid Overwatering – Water at the base instead of overhead to keep leaves dry.",
    "Bacterial_spot": "✅ Seed Treatment – Soak seeds in hot water (122°F for 25 minutes) to kill bacteria.\n✅ Copper Sprays – Apply copper-based fungicides to slow disease spread.\n✅ Sanitize Tools – Disinfect gardening tools to avoid spreading bacteria.\n✅ Remove Infected Leaves – Prune infected parts immediately and dispose of them.\n✅ Drip Irrigation – Water at the soil level to avoid leaf wetness.",
    "Late_blight": "✅ Disease-Free Seeds – Only plant certified, disease-free seeds or seedlings.\n✅ Fungicide Application – Apply fungicides containing chlorothalonil or copper.\n✅ Avoid Humid Conditions – Space plants properly to improve air circulation.\n✅ Destroy Infected Plants – Remove and burn infected plants to prevent spread.\n✅ Stake Plants – Keep plants off the ground to reduce moisture retention.",
    "Leaf_Mold": "✅ Grow Resistant Varieties – Use resistant tomato varieties like ‘Trust’ or ‘Red Sun’.\n✅ Greenhouse Ventilation – If growing indoors, ensure proper air circulation.\n✅ Reduce Humidity – Avoid overhead watering and provide sufficient spacing.\n✅ Apply Sulfur Fungicides – Use sulfur-based sprays to prevent the disease.\n✅ Remove Debris – Clear plant debris after harvesting to prevent overwintering fungi.",
    "Septoria_leaf_spot": "✅ Plant Rotation – Avoid growing tomatoes in the same location year after year.\n✅ Copper Fungicides – Apply fungicides with copper or chlorothalonil.\n✅ Keep Foliage Dry – Water in the morning and avoid wetting the leaves.\n✅ Remove Weeds – Eliminate nearby weeds, as they may harbor the fungus.\n✅ Use Mulch – Apply straw or plastic mulch to prevent soil splash contamination.",
    "Spider_mites Two-spotted_spider_mite": "✅ Introduce Natural Predators – Release ladybugs or predatory mites to control them.\n✅ Neem Oil Spray – Apply neem oil regularly to repel spider mites.\n✅ Increase Humidity – Spider mites thrive in dry conditions, so mist the plants.\n✅ Hose Down Plants – Use water sprays to wash off mites.\n✅ Remove Infested Leaves – Prune and discard affected leaves to prevent spreading.",
    "Target_Spot": "✅ Apply Fungicides – Use mancozeb or chlorothalonil-based fungicides.\n✅ Avoid Overcrowding – Space plants properly to allow airflow.\n✅ Monitor for Early Symptoms – Remove infected leaves immediately.\n✅ Improve Drainage – Prevent waterlogging by improving soil drainage.\n✅ Use Disease-Resistant Varieties – Some hybrid tomato varieties show resistance.",
    "Tomato_Yellow_Leaf_Curl_Virus": "✅ Control Whiteflies – TYLCV is spread by whiteflies, so use insecticides or yellow sticky traps.\n✅ Grow Resistant Varieties – Choose TYLCV-resistant tomato varieties.\n✅ Use Row Covers – Protect plants with floating row covers to prevent whitefly infestation.\n✅ Remove Infected Plants – Immediately discard infected plants to stop the spread.\n✅ Weed Management – Remove weeds that act as alternative hosts for whiteflies.",
    "Tomato_mosaic_virus": "✅ Use Disease-Free Seeds – Ensure seeds are certified virus-free.\n✅ Sanitize Hands and Tools – Wash hands and sterilize tools before handling plants.\n✅ Resistant Varieties – Grow ToMV-resistant tomato varieties.\n✅ Avoid Tobacco Products – Tobacco can carry the virus, so avoid handling plants after smoking.\n✅ Control Aphids – Aphids spread the virus, so use insecticidal soap or neem oil.",
    "Tomato__healthy": "✅ Rotate Crops – Change planting locations each season to prevent soil-borne diseases.\n✅ Use Organic Mulch – Retain moisture and reduce disease spread.\n✅ Regular Pruning – Remove lower leaves to improve airflow.\n✅ Balanced Fertilization – Avoid excess nitrogen, which makes plants more disease-prone.\n✅ Monitor & Act Early – Inspect plants frequently for any disease symptoms."
}

# Define a function to check if the uploaded file is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to preprocess the image before making predictions
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))  # Resize to model's expected input size
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize image
    return img_array

# Function to predict class and return details
def predict_image(img_array):
    # Predict the class probabilities
    prediction = model.predict(img_array)
    predicted_class_idx = np.argmax(prediction, axis=1)[0]
    confidence = np.max(prediction)

    # Confidence threshold
    CONFIDENCE_THRESHOLD = 0.7  # 70%

    # Load class indices
    with open('class_indices.json', 'r') as f:
        class_indices = json.load(f)

    # Reverse class indices to map index to class name
    class_labels = {v: k for k, v in class_indices.items()}
    predicted_class = class_labels.get(predicted_class_idx, "Unknown")

    # Check confidence threshold
    if confidence < CONFIDENCE_THRESHOLD:
        predicted_class = "Unknown"
        prevention = "This is not a valid tomato leaf image. Please upload a proper image of a tomato leaf."
    else:
        # Get prevention advice
        prevention = PREVENTION.get(predicted_class, "No prevention available.")

    return predicted_class, confidence, prevention

# Route for the homepage
@app.route('/')
def home():
    return render_template('index.html')

# Route for handling image uploads and predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400

        if file and allowed_file(file.filename):
            # Save the file
            filename = secure_filename(file.filename)
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(img_path)

            # Preprocess and predict
            img_array = preprocess_image(img_path)
            predicted_class, confidence, prevention = predict_image(img_array)

            # Clean up: delete the uploaded file
            os.remove(img_path)

            # Return prediction, confidence, and prevention advice
            return jsonify({
                "predicted_class": predicted_class,
                "confidence": round(confidence * 100, 2),  # Confidence in percentage
                "prevention": prevention
            })
        else:
            return jsonify({"error": "Invalid file type"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Start the Flask server
if __name__ == '__main__':
    app.run(debug=True)
