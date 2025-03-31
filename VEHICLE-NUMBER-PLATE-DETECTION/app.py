# Import necessary dependencies
from flask import Flask, render_template, request, send_file
from flask_cors import CORS, cross_origin
import numpy as np
import os
import tensorflow as tf
import easyocr as ocr
import cv2
from object_detection.utils import label_map_util

# Setting environment variables
os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

app = Flask(__name__)  # Initializing a Flask app
CORS(app)

# Model Paths
MODEL_NAME = 'auto_no_plate_detection_model'
PATH_TO_FROZEN_GRAPH = os.path.join(MODEL_NAME, 'frozen_inference_graph.pb')
PATH_TO_LABELS = os.path.join('training', 'classes.pbtxt')

# Load TensorFlow Graph
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.io.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

# Load label map
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

# OCR Reader
reader = ocr.Reader(['en'])

# State Code Mapping
states = {
    "AN": "Andaman and Nicobar", "AP": "Andhra Pradesh", "AR": "Arunachal Pradesh", "AS": "Assam",
    "BR": "Bihar", "CH": "Chandigarh", "DN": "Dadra and Nagar Haveli", "DD": "Daman and Diu",
    "DL": "Delhi", "GA": "Goa", "GJ": "Gujarat", "HR": "Haryana", "HP": "Himachal Pradesh",
    "JK": "Jammu and Kashmir", "KA": "Karnataka", "KL": "Kerala", "LD": "Lakshadweep",
    "MP": "Madhya Pradesh", "MH": "Maharashtra", "MN": "Manipur", "ML": "Meghalaya",
    "MZ": "Mizoram", "NL": "Nagaland", "OD": "Odissa", "PY": "Pondicherry", "PN": "Punjab",
    "RJ": "Rajasthan", "SK": "Sikkim", "TN": "Tamil Nadu", "TR": "Tripura", "UP": "Uttar Pradesh",
    "WB": "West Bengal", "CG": "Chhattisgarh", "TS": "Telangana", "JH": "Jharkhand", "UK": "Uttarakhand"
}

@app.route('/', methods=['GET'])
@cross_origin()
def homePage():
    return render_template("index.html")

@app.route('/startapp', methods=['POST', 'GET'])
@cross_origin()
def upload_img():
    if request.method == 'POST':
        return render_template('upload.html')
    return "Something went wrong"

@app.route('/detection', methods=['POST'])
@cross_origin()
def detection():
    try:
        # Get uploaded file
        uploaded_file = request.files.get('upload_file')
        if not uploaded_file:
            return "Error: No file uploaded"

        filename = uploaded_file.filename
        print(f"Uploaded File: {filename}")

        # Save the file
        uploaded_file.save(filename)

        # Check valid image format
        allowed_extensions = ["JPG", "JPEG", "PNG"]
        extension = filename.split(".")[-1].upper()
        if extension not in allowed_extensions:
            return 'Error: Please upload a JPG, JPEG, or PNG file'

        # Load image
        try:
            image_np = cv2.imread(filename, 1)
            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
            image_np = cv2.resize(image_np, (800, 600))
        except Exception as e:
            return f"Error: Unable to process image file ({str(e)})"

        # Clear previous detections
        delete_folder = './detected_number_plate'
        if os.path.exists(delete_folder):
            for file in os.listdir(delete_folder):
                os.remove(os.path.join(delete_folder, file))

        # Object Detection
        with detection_graph.as_default():
            with tf.compat.v1.Session(graph=detection_graph) as sess:
                image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                scores = detection_graph.get_tensor_by_name('detection_scores:0')
                classes = detection_graph.get_tensor_by_name('detection_classes:0')

                image_np_expanded = np.expand_dims(image_np, axis=0)
                (boxes, scores, classes, _) = sess.run(
                    [boxes, scores, classes, detection_graph.get_tensor_by_name('num_detections:0')],
                    feed_dict={image_tensor: image_np_expanded})

                # Process detections
                img_height, img_width, _ = image_np.shape
                min_score_thresh = 0.5

                for i, score in enumerate(scores[0]):
                    if score > min_score_thresh:
                        ymin, xmin, ymax, xmax = boxes[0][i]
                        ymin, xmin, ymax, xmax = int(ymin * img_height), int(xmin * img_width), int(ymax * img_height), int(xmax * img_width)

                        # Expand bounding box for better OCR accuracy
                        ymin, xmin = max(0, ymin - 20), max(0, xmin - 20)
                        ymax, xmax = min(img_height, ymax + 30), min(img_width, xmax + 30)

                        # Crop detected plate
                        roi = image_np[ymin:ymax, xmin:xmax]
                        if roi.size == 0:
                            return "Error: Cropped image is empty!"

                        # Save detected plate
                        detected_path = os.path.join(delete_folder, "plate.png")
                        cv2.imwrite(detected_path, roi)

                        print("Detected number plate saved successfully")

                        # OCR Processing
                        output = reader.readtext(detected_path)
                        if not output:
                            return "Error: No text detected on number plate"
                        
                        final_output = output[0][1].replace(" ", "").upper()
                        print("Extracted Text:", final_output)

                        # Extract State Code
                        state_code = final_output[:2]
                        state_name = states.get(state_code, "Unknown State")

                        # Display results
                        return render_template('show_map.html', image_name1=f"./static/maps/{state_code}.jpg", state_name=f"{state_name} :- {final_output}")

        return "Error: No number plate detected"

    except Exception as e:
        print("Error:", e)
        return f"Error: {str(e)}"

@app.route('/uploadfile', methods=['GET'])
@cross_origin()
def uploadfile():
    return render_template('upload.html')

if __name__ == "__main__":
    app.run(host='127.0.0.1', port=8000, debug=True)
