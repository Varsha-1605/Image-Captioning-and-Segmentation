# # # web_app.py
# # import os
# # from flask import Flask, render_template, request, redirect, url_for, flash
# # from werkzeug.utils import secure_filename
# # import sys
# # import shutil


# # # Add the 'src' directory to Python's path so we can import from it.
# # sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

# # # IMPORTANT: Explicitly import the CLASSES directly into the __main__ scope.
# # # This ensures their definitions are available to torch.load when it attempts
# # # to deserialize pickled objects (like COCOVocabulary or ImageCaptioningModel
# # # instances) that might have been saved with a __main__ module reference.
# # from src.data_preprocessing import COCOVocabulary
# # from src.model import ImageCaptioningModel

# # # Now import the necessary functions and modules from your project
# # from src.inference_api import generate_caption_for_image
# # from src.utils import get_logger

# # # Initialize Flask app
# # app = Flask(__name__)

# # logger = get_logger(__name__)

# # # --- Configuration for Flask App ---
# # # Define the folder to store uploaded images temporarily within the static directory
# # UPLOAD_FOLDER = os.path.join('static', 'uploads') # This makes 'uploads' a subfolder of 'static'
# # ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# # app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# # app.config['SECRET_KEY'] = 'your_super_secret_key_change_this_in_production' # <--- CHANGE THIS!

# # # Create the upload folder if it doesn't exist
# # # Ensure the full path is created, relative to the app's root.
# # os.makedirs(os.path.join(app.root_path, UPLOAD_FOLDER), exist_ok=True)
# # logger.info(f"Upload folder '{UPLOAD_FOLDER}' ensured.")

# # def allowed_file(filename):
# #     return '.' in filename and \
# #            filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# # @app.route('/')
# # def index():
# #     """
# #     Renders the main page with the image upload form.
# #     """
# #     # Initialize with None for variables that might not be present on first load
# #     return render_template('index.html', caption=None, uploaded_image_url=None)

# # @app.route('/predict', methods=['POST'])
# # def predict():
# #     """
# #     Handles image upload, generates caption, and displays results.
# #     """
# #     logger.info("Received request to /predict.")

# #     if 'file' not in request.files:
# #         flash('No file part', 'error') # Added 'error' category for red flash message
# #         logger.warning("No file part in the request.")
# #         return redirect(request.url)

# #     file = request.files['file']

# #     if file.filename == '':
# #         flash('No selected file', 'error') # Added 'error' category
# #         logger.warning("No file selected by user.")
# #         return redirect(request.url)

# #     uploaded_image_url = None # Initialize to None

# #     if file and allowed_file(file.filename):
# #         filename = secure_filename(file.filename)
# #         # Construct the full file path where the image will be saved
# #         filepath = os.path.join(app.root_path, app.config['UPLOAD_FOLDER'], filename)
# #         file.save(filepath) # Save the uploaded file to the static/uploads folder
# #         logger.info(f"File saved to: {filepath}")

# #         # Construct the URL for the saved image, accessible via Flask's static files
# #         uploaded_image_url = url_for('static', filename='uploads/' + filename)

# #         generated_caption = "Error: Could not generate caption."
# #         try:
# #             generated_caption = generate_caption_for_image(filepath)
# #             logger.info(f"Caption generated: {generated_caption}")
# #             flash('Caption generated successfully!', 'success') # Added 'success' category for green flash message
# #         except FileNotFoundError:
# #             flash(f"Error: Model or image file not found. Check server logs.", 'error')
# #             logger.error(f"Model or image file not found during inference for {filepath}.")
# #         except RuntimeError as re:
# #             flash(f"Error: Model not initialized. Check server logs.", 'error')
# #             logger.critical(f"Model not initialized: {re}", exc_info=True)
# #         except Exception as e:
# #             flash(f"An error occurred during caption generation. Check server logs.", 'error')
# #             logger.critical(f"Error generating caption for {filepath}: {e}", exc_info=True)
# #         finally:
# #             # IMPORTANT: We are NOT deleting the file here, as it needs to be served by Flask
# #             # for display. Implement a separate cleanup mechanism for old files if needed.
# #             pass # Keep the file in static/uploads for display

# #         # Render the template again, passing the generated caption and the URL to the uploaded image
# #         return render_template('index.html', caption=generated_caption, uploaded_image_url=uploaded_image_url)
# #     else:
# #         flash('Allowed image types are png, jpg, jpeg, gif', 'error') # Added 'error' category
# #         logger.warning(f"Disallowed file type uploaded: {file.filename}")
# #         return redirect(request.url)

# # if __name__ == '__main__':
# #     logger.info("Starting Flask web application...")
# #     app.run(debug=True)









# import os
# from flask import Flask, render_template, request, redirect, url_for, flash
# from werkzeug.utils import secure_filename
# import sys
# import shutil
# import torch
# from PIL import Image
# import numpy as np
# import cv2 # For image manipulation and plotting segmentation results
# # import pickle # Not directly needed in web_app.py if generate_caption_for_image handles it
# # import json # Not directly needed in web_app.py for basic metrics, but good to keep

# # Add the 'src' directory to Python's path so we can import from it.
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

# # IMPORTANT: Explicitly import the CLASSES directly into the __main__ scope.
# # This ensures their definitions are available to torch.load when it attempts
# # to deserialize pickled objects (like COCOVocabulary or ImageCaptioningModel
# # instances) that might have been saved with a __main__ module reference.
# # These imports are here because your original working web_app.py had them.
# from src.data_preprocessing import COCOVocabulary
# from src.model import ImageCaptioningModel

# # Now import the necessary functions and modules from your project
# from src.inference_api import generate_caption_for_image # Your existing captioning function
# from src.utils import get_logger # Your existing logger utility

# # Import YOLO for segmentation - adapted from your file.py
# try:
#     from ultralytics import YOLO
#     logger = get_logger(__name__) # Initialize logger here after imports
# except ImportError:
#     logger = get_logger(__name__)
#     logger.error("ultralytics library not found. Please install it: pip install ultralytics")
#     YOLO = None # Set to None if import fails


# # Initialize Flask app
# app = Flask(__name__)

# # --- Configuration for Flask App ---
# # Define the folder to store uploaded images temporarily within the static directory
# UPLOAD_FOLDER = os.path.join('static', 'uploads') # This makes 'uploads' a subfolder of 'static'
# ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# app.config['SECRET_KEY'] = 'your_super_secret_key_change_this_in_production' # <--- CHANGE THIS!

# # Create the upload folder if it doesn't exist
# # Ensure the full path is created, relative to the app's root.
# os.makedirs(os.path.join(app.root_path, UPLOAD_FOLDER), exist_ok=True)
# logger.info(f"Upload folder '{UPLOAD_FOLDER}' ensured at {os.path.join(app.root_path, UPLOAD_FOLDER)}")

# # --- Global Segmentation Model Loading ---
# # Captioning model is assumed to be loaded/handled by generate_caption_for_image internally.
# segmentation_model_yolo = None # Renamed to avoid conflict with 'model' in calculate_metrics (from file.py context)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# logger.info(f"Using device for models: {device}")

# try:
#     if YOLO: # Only try to load if ultralytics import was successful
#         segmentation_model_yolo = YOLO('yolov8x-seg.pt') # YOLOv8x-seg is a large segmentation model
#         segmentation_model_yolo.to(device) # Move model to appropriate device
#         logger.info("Segmentation Model (YOLOv8x-seg) loaded successfully.")
#     else:
#         logger.warning("YOLO library not available, skipping segmentation model loading.")
# except Exception as e:
#     logger.critical(f"Error loading Segmentation Model (YOLOv8x-seg): {e}", exc_info=True)
#     segmentation_model_yolo = None


# # --- Segmentation Helper Functions (Adapted from your file.py) ---
# # These functions are kept here as they were adapted from file.py to fit Flask.

# def validate_image_for_seg(img_array):
#     """Ensure 3-channel RGB format for OpenCV processing."""
#     if len(img_array.shape) == 2:  # Grayscale
#         img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
#     elif img_array.shape[2] == 4:  # RGBA
#         img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
#     elif img_array.shape[2] > 3:  # Extra channels
#         img_array = img_array[:, :, :3]
#     return img_array

# def calculate_segmentation_metrics(results, segmentation_model_ref):
#     """
#     Calculates basic segmentation metrics (detected objects).
#     Adapted to remove Streamlit dependencies and reliance on mock GT.
#     """
#     metrics = {
#         'detected_objects': [],
#         'num_objects': 0,
#         'status': 'Processed',
#         'error': None
#     }

#     if not segmentation_model_ref:
#         metrics['error'] = "Segmentation model not loaded."
#         metrics['status'] = "Error: Segmentation model unavailable."
#         return metrics
    
#     if not results or results[0].masks is None or len(results[0].masks) == 0:
#         metrics['status'] = "No objects detected."
#         return metrics
    
#     try:
#         detected_objects_info = []
#         for r_box in results[0].boxes.data.tolist():
#             class_id = int(r_box[5])
#             confidence = round(r_box[4], 2)
#             # Ensure class_id exists in model.names
#             class_name = segmentation_model_ref.names.get(class_id, f"Class {class_id}")
#             detected_objects_info.append(f"{class_name} (Conf: {confidence})")
        
#         metrics['detected_objects'] = detected_objects_info
#         metrics['num_objects'] = len(detected_objects_info)
        
#     except Exception as e:
#         metrics['error'] = f"Metric calculation failed: {str(e)}"
#         metrics['status'] = "Error during metric calculation."
#         logger.error(f"Metric calculation failed: {e}", exc_info=True)
        
#     return metrics

# def perform_segmentation(image_path, model_ref, upload_folder, filename_stem):
#     """
#     Performs segmentation on an image and returns the URL of the segmented image
#     and a dictionary of metrics.
#     """
#     segmented_image_url = None
#     metrics = {}

#     if not model_ref:
#         metrics = {'error': "Segmentation model not loaded."}
#         return segmented_image_url, metrics

#     try:
#         img_pil = Image.open(image_path).convert('RGB')
#         img_np = np.array(img_pil) # Convert to NumPy array
#         img_cv2 = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR) # YOLO expects BGR

#         # Perform inference
#         results = model_ref(img_cv2, verbose=False) # verbose=False suppresses console output
        
#         if results and results[0].masks is not None and len(results[0].masks) > 0:
#             # Plot results directly onto the image
#             annotated_image = results[0].plot() # This returns a numpy array (BGR)
            
#             # Convert BGR (OpenCV default) to RGB for PIL and saving
#             annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
#             img_segmented_pil = Image.fromarray(annotated_image_rgb)

#             # Save the segmented image
#             segmented_filename = f"segmented_{filename_stem}.png" # Ensure .png extension for segmented output
#             segmented_filepath = os.path.join(upload_folder, segmented_filename)
#             img_segmented_pil.save(segmented_filepath)
#             segmented_image_url = url_for('static', filename=f'uploads/{segmented_filename}')
#             logger.info(f"Segmented image saved to: {segmented_filepath}")

#             # Calculate and return metrics
#             metrics = calculate_segmentation_metrics(results, model_ref)
#             metrics['status'] = "Segmentation successful."
#         else:
#             metrics['status'] = "No objects detected for segmentation."
#             logger.info(f"No objects detected for segmentation in {image_path}.")

#     except Exception as e:
#         metrics['error'] = str(e)
#         metrics['status'] = "Error during segmentation processing."
#         logger.critical(f"Error in perform_segmentation for {image_path}: {e}", exc_info=True)
        
#     return segmented_image_url, metrics


# def allowed_file(filename):
#     return '.' in filename and \
#            filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# @app.route('/')
# def index():
#     """
#     Renders the main page.
#     Initializes Flask variables for the template to None/empty dict.
#     """
#     return render_template('index.html', 
#                            caption=None, 
#                            uploaded_image_url=None,
#                            segmentation_image_url=None,
#                            segmentation_metrics={})

# @app.route('/predict', methods=['POST'])
# def predict():
#     """
#     Handles image upload, performs captioning and segmentation,
#     and renders the results.
#     """
#     logger.info("Received request to /predict.")

#     # Initialize variables for the template
#     generated_caption = "N/A"
#     uploaded_image_url = None
#     segmentation_image_url = None
#     segmentation_metrics = {} 

#     if 'file' not in request.files:
#         flash('No file part in the request.', 'error')
#         logger.warning("No file part.")
#         return redirect(request.url)
    
#     file = request.files['file']
#     if file.filename == '':
#         flash('No selected file.', 'error')
#         logger.warning("Empty filename.")
#         return redirect(request.url)
    
#     if file and allowed_file(file.filename):
#         filename = secure_filename(file.filename)
#         # Extract filename stem (without extension) for segmented image naming
#         filename_stem, file_ext = os.path.splitext(filename)
        
#         # Construct the full path to save the original image
#         original_filepath = os.path.join(app.root_path, app.config['UPLOAD_FOLDER'], filename)
#         file.save(original_filepath)
#         uploaded_image_url = url_for('static', filename=f'uploads/{filename}')
#         logger.info(f"Original image saved to: {original_filepath}")
        
#         # --- Perform Image Captioning (using your provided correct method) ---
#         try:
#             logger.info(f"Starting caption generation for {original_filepath}...")
#             # This is your original working call for captioning
#             generated_caption = generate_caption_for_image(original_filepath)
#             logger.info(f"Caption generated: '{generated_caption}'")
#             flash("Image caption generated successfully!", 'success')
#         except FileNotFoundError:
#             flash(f"Error: Captioning model or vocabulary file not found for {original_filepath}. Check server logs.", 'error')
#             logger.error(f"Captioning model/vocab not found during inference for {original_filepath}.")
#             generated_caption = "Error: Captioning model/vocab not found."
#         except RuntimeError as re:
#             flash(f"Error: Captioning model not initialized or device issue. Check server logs.", 'error')
#             logger.critical(f"Captioning model not initialized: {re}", exc_info=True)
#             generated_caption = "Error: Captioning service unavailable."
#         except Exception as e:
#             flash(f"An unexpected error occurred during caption generation. Check server logs.", 'error')
#             logger.critical(f"Error generating caption for {original_filepath}: {e}", exc_info=True)
#             generated_caption = "Error: Could not generate caption."


#         # --- Perform Image Segmentation ---
#         if segmentation_model_yolo:
#             logger.info(f"Starting segmentation for {original_filepath} using YOLO...")
#             segmentation_image_url, segmentation_metrics = perform_segmentation(
#                 image_path=original_filepath,
#                 model_ref=segmentation_model_yolo,
#                 upload_folder=os.path.join(app.root_path, app.config['UPLOAD_FOLDER']),
#                 filename_stem=filename_stem
#             )
#             if 'error' in segmentation_metrics and segmentation_metrics['error']:
#                  flash(f"Segmentation Error: {segmentation_metrics['error']}", 'error')
#             elif segmentation_image_url:
#                 flash("Image segmentation performed successfully!", 'success')
#             else:
#                 flash(f"Segmentation: {segmentation_metrics.get('status', 'No specific status.')}", 'info')
#         else:
#             flash("Segmentation model not initialized. Ensure 'ultralytics' is installed and model loaded.", 'error')
#             logger.error("Segmentation model (YOLO) is not available.")
#             segmentation_metrics['error'] = "Segmentation service unavailable."


#         # Render the template with results for both tasks
#         return render_template('index.html', 
#                                caption=generated_caption, 
#                                uploaded_image_url=uploaded_image_url,
#                                segmentation_image_url=segmentation_image_url,
#                                segmentation_metrics=segmentation_metrics)
#     else:
#         flash('Allowed image types are png, jpg, jpeg, gif.', 'error')
#         logger.warning(f"Disallowed file type uploaded: {file.filename}")
#         return redirect(request.url)

# if __name__ == '__main__':
#     logger.info("Starting Flask web application...")
#     # In production, use a more robust WSGI server like Gunicorn or uWSGI
#     app.run(debug=True, host='0.0.0.0', port=5000) # Listen on all interfaces and port 5000













































import os
import base64
import json
import numpy as np
import cv2 # For image manipulation and plotting segmentation results
import pickle # To load vocabulary (for captioning internal loading)
import logging
from io import BytesIO
import sys 
# import jsonify # This was the missing import causing issues!
import face_recognition # For facial recognition tasks
import torch
from werkzeug.utils import secure_filename


from flask import Flask, render_template, request, redirect, url_for, session, flash, g, jsonify # Corrected import: added jsonify
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
from PIL import Image # Used for both image processing and face_recognition

# Add the 'src' directory to Python's path so we can import from it.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

# IMPORTANT: Explicitly import the CLASSES directly into thecls __main__ scope.
# This ensures their definitions are available to torch.load when it attempts
# to deserialize pickled objects (like COCOVocabulary or ImageCaptioningModel
# instances) that might have been saved with a __main__ module reference.
from src.data_preprocessing import COCOVocabulary
from src.model import ImageCaptioningModel

# Now import the necessary functions and modules from your project
from src.inference_api import generate_caption_for_image # Your existing captioning function
from src.utils import get_logger # Your existing logger utility

# Import YOLO for segmentation - adapted from your file.py
try:
    from ultralytics import YOLO
    # Logger initialization moved here to ensure it's after all necessary imports
    logger = get_logger(__name__) 
except ImportError:
    logger = get_logger(__name__)
    logger.error("ultralytics library not found. Please install it: pip install ultralytics")
    YOLO = None # Set to None if import fails

# --- Flask App Setup ---
app = Flask(__name__)

# --- Configuration ---
# Strong secret key for session management (IMPORTANT: Change this in production!)
app.config['SECRET_KEY'] = os.urandom(24) 
# SQLite database for users
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db' 
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Define the folder to store uploaded images temporarily within the static directory
UPLOAD_FOLDER = os.path.join('static', 'uploads') 
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
os.makedirs(os.path.join(app.root_path, UPLOAD_FOLDER), exist_ok=True)
logger.info(f"Upload folder '{UPLOAD_FOLDER}' ensured at {os.path.join(app.root_path, UPLOAD_FOLDER)}")

# --- Database Initialization ---
db = SQLAlchemy(app)

# Suppress some logging for cleaner output, but keep our custom prints
logging.getLogger("werkzeug").setLevel(logging.ERROR)

# --- Database Model (from auth_app.py) ---
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    # Store face encodings as JSON string of a list of floats (numpy arrays are not directly JSON serializable)
    face_encodings_json = db.Column(db.Text, nullable=True) 

    def __repr__(self):
        return f'<User {self.email}>'

# Create database tables if they don't exist
with app.app_context():
    db.create_all()
    print("Database tables created/checked.") # Kept print from your auth_app.py

# --- Global Segmentation Model Loading ---
# Captioning model is assumed to be loaded/handled by generate_caption_for_image internally.
segmentation_model_yolo = None 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device for models: {device}")

try:
    if YOLO: # Only try to load if ultralytics import was successful
        segmentation_model_yolo = YOLO('yolov8x-seg.pt') # YOLOv8x-seg is a large segmentation model
        segmentation_model_yolo.to(device) # Move model to appropriate device
        logger.info("Segmentation Model (YOLOv8x-seg) loaded successfully.")
    else:
        logger.warning("YOLO library not available, skipping segmentation model loading.")
except Exception as e:
    logger.critical(f"Error loading Segmentation Model (YOLOv8x-seg): {e}", exc_info=True)
    segmentation_model_yolo = None

# --- Helper Functions for Facial Recognition (Copied directly from your working auth_app.py) ---
def get_face_encoding_from_image(image_data_b64):
    """
    Decodes base64 image data, finds faces, and returns the first face's encoding.
    Returns None if no face is found or on error.
    """
    try:
        print(f"Processing image data of length: {len(image_data_b64)}") # From your auth_app.py
        
        # Handle both formats: with and without data URL prefix
        if ',' in image_data_b64:
            # Remove data URL prefix (e.g., "data:image/jpeg;base64,")
            image_data_clean = image_data_b64.split(',')[1]
        else:
            image_data_clean = image_data_b64
        
        # Add padding if needed (base64 strings must be multiples of 4)
        missing_padding = len(image_data_clean) % 4
        if missing_padding:
            image_data_clean += '=' * (4 - missing_padding)
            
        try:
            image_bytes = base64.b64decode(image_data_clean)
        except Exception as decode_error:
            print(f"Base64 decode error: {decode_error}") # From your auth_app.py
            return None
            
        print(f"Decoded image bytes length: {len(image_bytes)}") # From your auth_app.py
        
        # Open and convert image
        try:
            img = Image.open(BytesIO(image_bytes))
            print(f"Image opened successfully. Format: {img.format}, Size: {img.size}, Mode: {img.mode}") # From your auth_app.py
            
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
                print(f"Converted image to RGB mode") # From your auth_app.py
            
        except Exception as img_error:
            print(f"Image opening/conversion error: {img_error}") # From your auth_app.py
            return None
        
        # Convert PIL Image to numpy array (face_recognition expects numpy array)
        img_np = np.array(img)
        print(f"Numpy array shape: {img_np.shape}") # From your auth_app.py
        
        # Find face encodings
        try:
            face_locations = face_recognition.face_locations(img_np)
            print(f"Found {len(face_locations)} face location(s)") # From your auth_app.py
            
            if len(face_locations) == 0:
                print("No faces detected in the image") # From your auth_app.py
                return None
            
            face_encodings = face_recognition.face_encodings(img_np, face_locations)
            print(f"Generated {len(face_encodings)} face encoding(s)") # From your auth_app.py
            
            if len(face_encodings) > 0:
                encoding = face_encodings[0]
                print(f"Face encoding shape: {encoding.shape}") # From your auth_app.py
                return encoding.tolist() # Convert numpy array to list for JSON serialization
            else:
                print("No face encodings generated despite face locations found") # From your auth_app.py
                return None
                
        except Exception as face_error:
            print(f"Face recognition processing error: {face_error}") # From your auth_app.py
            return None
            
    except Exception as e:
        print(f"General error processing image for face encoding: {e}") # From your auth_app.py
        return None

def compare_face_encoding_to_stored(live_encoding, stored_encodings_json):
    """
    Compares a live face encoding to a list of stored encodings for a user.
    Returns True if a match is found, False otherwise.
    """
    if not live_encoding:
        print("Live encoding is None, cannot compare.") # From your auth_app.py
        return False
    if not stored_encodings_json:
        print("Stored encodings JSON is None, cannot compare.") # From your auth_app.py
        return False
    
    try:
        # Convert JSON string back to a list of numpy arrays
        stored_encodings_list = json.loads(stored_encodings_json)
        if not stored_encodings_list:
            print("No stored encodings found in JSON") # From your auth_app.py
            return False
            
        stored_encodings = [np.array(e) for e in stored_encodings_list]
        print(f"Comparing against {len(stored_encodings)} stored encodings") # From your auth_app.py
        
        # Compare the live encoding against all stored encodings for this user
        # tolerance: lower value means stricter match (0.6 is common default)
        matches = face_recognition.compare_faces(stored_encodings, np.array(live_encoding), tolerance=0.6)
        
        match_found = True in matches
        print(f"Face comparison result: {match_found}. Matches: {matches}") # From your auth_app.py
        
        return match_found
    except Exception as e:
        print(f"Error comparing face encodings: {e}") # From your auth_app.py
        return False

# --- Segmentation Helper Functions (from web_app.py, adapted from file.py) ---
def calculate_segmentation_metrics(results, segmentation_model_ref):
    """
    Calculates basic segmentation metrics (detected objects).
    Adapted to remove Streamlit dependencies and reliance on mock GT.
    """
    metrics = {
        'detected_objects': [],
        'num_objects': 0,
        'status': 'Processed',
        'error': None
    }

    if not segmentation_model_ref:
        metrics['error'] = "Segmentation model not loaded."
        metrics['status'] = "Error: Segmentation model unavailable."
        return metrics
    
    if not results or results[0].masks is None or len(results[0].masks) == 0:
        metrics['status'] = "No objects detected."
        return metrics
    
    try:
        detected_objects_info = []
        for r_box in results[0].boxes.data.tolist():
            class_id = int(r_box[5])
            confidence = round(r_box[4], 2)
            # Ensure class_id exists in model.names
            class_name = segmentation_model_ref.names.get(class_id, f"Class {class_id}")
            detected_objects_info.append(f"{class_name} (Conf: {confidence})")
        
        metrics['detected_objects'] = detected_objects_info
        metrics['num_objects'] = len(detected_objects_info)
        
    except Exception as e:
        metrics['error'] = f"Metric calculation failed: {str(e)}"
        metrics['status'] = "Error during metric calculation."
        logger.error(f"Metric calculation failed: {e}", exc_info=True)
        
    return metrics

def perform_segmentation(image_path, model_ref, upload_folder, filename_stem):
    """
    Performs segmentation on an image and returns the URL of the segmented image
    and a dictionary of metrics.
    """
    segmented_image_url = None
    metrics = {}

    if not model_ref:
        metrics = {'error': "Segmentation model not loaded."}
        return segmented_image_url, metrics

    try:
        img_pil = Image.open(image_path).convert('RGB')
        img_np = np.array(img_pil) # Convert to NumPy array
        img_cv2 = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR) # YOLO expects BGR

        # Perform inference
        results = model_ref(img_cv2, verbose=False) # verbose=False suppresses console output
        
        if results and results[0].masks is not None and len(results[0].masks) > 0:
            # Plot results directly onto the image
            annotated_image = results[0].plot() # This returns a numpy array (BGR)
            
            # Convert BGR (OpenCV default) to RGB for PIL and saving
            annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
            img_segmented_pil = Image.fromarray(annotated_image_rgb)

            # Save the segmented image
            segmented_filename = f"segmented_{filename_stem}.png" # Ensure .png extension for segmented output
            segmented_filepath = os.path.join(upload_folder, segmented_filename)
            img_segmented_pil.save(segmented_filepath)
            segmented_image_url = url_for('static', filename=f'uploads/{segmented_filename}')
            logger.info(f"Segmented image saved to: {segmented_filepath}")

            # Calculate and return metrics
            metrics = calculate_segmentation_metrics(results, model_ref)
            metrics['status'] = "Segmentation successful."
        else:
            metrics['status'] = "No objects detected for segmentation."
            logger.info(f"No objects detected for segmentation in {image_path}.")

    except Exception as e:
        metrics['error'] = str(e)
        metrics['status'] = "Error during segmentation processing."
        logger.critical(f"Error in perform_segmentation for {image_path}: {e}", exc_info=True)
        
    return segmented_image_url, metrics

# --- Helper for file extension check ---
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- Before/After Request Hooks ---
@app.before_request
def load_logged_in_user():
    user_id = session.get('user_id')
    if user_id is None:
        g.user = None
    else:
        g.user = User.query.get(user_id)

# --- Authentication Decorator ---
def login_required(view):
    @wraps(view)
    def wrapped_view(**kwargs):
        if g.user is None:
            flash("Please log in to access this page.", "info")
            return redirect(url_for('auth_page'))
        return view(**kwargs)
    return wrapped_view


# --- Routes (Combined from both previous apps, authentication parts use print for logs) ---

# Authentication Page Route (Root)
@app.route('/')
def auth_page():
    """Serves the authentication HTML page. Redirects to main_app if logged in."""
    if 'user_id' in session: # Use session directly as in your original auth_app.py
        user = User.query.get(session['user_id'])
        if user:
            # If already logged in, redirect to main_app
            print(f"User {user.email} already logged in, redirecting to main_app.")
            return redirect(url_for('main_app'))
        else:
            session.pop('user_id', None) # Clear invalid session
            print("Invalid user_id in session, redirecting to auth_page.")
            return redirect(url_for('auth_page'))
    print("Serving auth.html for login/registration.")
    return render_template('auth.html')

@app.route('/register', methods=['POST'])
def register():
    """Handles traditional email/password registration."""
    email = request.form['email'].strip() # Use .strip() to remove whitespace
    password = request.form['password']
    print(f"Received traditional registration request for: {email}") # Kept print from your auth_app.py

    if not email or not password:
        print("Error: Email or password missing for traditional registration.") # From auth_app.py
        return jsonify({'success': False, 'message': 'Email and password are required.'}), 400

    import re # Make sure re is imported
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    if not re.match(email_pattern, email):
        print(f"Error: Invalid email format: {email}")
        return jsonify({'success': False, 'message': 'Please enter a valid email address.'}), 400

    if len(password) < 6:
        print("Error: Password too short.") # From auth_app.py
        return jsonify({'success': False, 'message': 'Password must be at least 6 characters long.'}), 400

    existing_user = User.query.filter_by(email=email).first()
    if existing_user:
        print(f"Error: Email {email} already registered.") # From auth_app.py
        return jsonify({'success': False, 'message': 'Email already registered.'}), 409

    hashed_password = generate_password_hash(password)
    new_user = User(email=email, password_hash=hashed_password)
    
    try:
        db.session.add(new_user)
        db.session.commit()
        print(f"Traditional registration successful for: {email}") # From auth_app.py
        return jsonify({'success': True, 'message': 'Registration successful. You can now log in.'})
    except Exception as e:
        db.session.rollback()
        print(f"Database error during traditional registration: {e}") # From auth_app.py
        return jsonify({'success': False, 'message': 'Database error during registration.'}), 500


@app.route('/login', methods=['POST'])
def login():
    """Handles traditional email/password login."""
    email = request.form['email'].strip() # Use .strip()
    password = request.form['password']
    print(f"Received traditional login request for: {email}") # From auth_app.py

    user = User.query.filter_by(email=email).first()

    if user and check_password_hash(user.password_hash, password):
        session['user_id'] = user.id
        print(f"Traditional login successful for: {email}") # From auth_app.py
        return jsonify({'success': True, 'message': 'Login successful.'})
    else:
        print(f"Traditional login failed for: {email}") # From auth_app.py
        return jsonify({'success': False, 'message': 'Invalid email or password.'}), 401

@app.route('/face_register', methods=['POST'])
def face_register():
    """
    Receives face images for registration.
    Expects email, password, and a list of base64 image data from the frontend.
    """
    try:
        if not request.is_json:
            print("Error: Request is not JSON for face_register") # From auth_app.py
            return jsonify({'success': False, 'message': 'Invalid request format. JSON expected.'}), 400
            
        data = request.get_json()
        if not data:
            print("Error: No JSON data received for face_register") # From auth_app.py
            return jsonify({'success': False, 'message': 'No data received.'}), 400
            
        email = data.get('email', '').strip() # Use .strip()
        password = data.get('password', '')
        image_data_list = data.get('images', [])
        
        print(f"Received face registration request for: {email} with {len(image_data_list)} images.") # From auth_app.py

        if not email or not password or not image_data_list:
            print("Error: Missing email, password or image data for face registration.") # From auth_app.py
            return jsonify({'success': False, 'message': 'Email, password, and face images are required.'}), 400

        import re # Make sure re is imported
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(email_pattern, email):
            print(f"Error: Invalid email format: {email}")
            return jsonify({'success': False, 'message': 'Please enter a valid email address.'}), 400

        if len(password) < 6:
            print("Error: Password too short for face registration.") # From auth_app.py
            return jsonify({'success': False, 'message': 'Password must be at least 6 characters long.'}), 400

        existing_user = User.query.filter_by(email=email).first()
        if existing_user:
            print(f"Error: Email {email} already registered for face registration.") # From auth_app.py
            return jsonify({'success': False, 'message': 'Email already registered.'}), 409

        all_encodings = []
        for i, img_b64 in enumerate(image_data_list):
            print(f"Processing image {i+1}/{len(image_data_list)} for face encoding...") # From auth_app.py
            
            if not img_b64:
                print(f"Warning: Image {i+1} is empty, skipping.") # From auth_app.py
                continue
                
            encoding = get_face_encoding_from_image(img_b64)
            if encoding:
                all_encodings.append(encoding)
                print(f"Successfully processed image {i+1}") # From auth_app.py
            else:
                print(f"Failed to process image {i+1} - no face detected or processing error.") # From auth_app.py
        
        if not all_encodings:
            print("Error: No detectable faces found in any of the provided images for face registration.") # From auth_app.py
            return jsonify({'success': False, 'message': 'No detectable faces in the provided images. Please try again with clearer images showing your face clearly.'}), 400

        hashed_password = generate_password_hash(password)
        face_encodings_json = json.dumps(all_encodings) 
        
        new_user = User(email=email, password_hash=hashed_password, face_encodings_json=face_encodings_json)
        
        try:
            db.session.add(new_user)
            db.session.commit()
            print(f"Face registration successful for: {email}. Stored {len(all_encodings)} encodings.") # From auth_app.py
            return jsonify({'success': True, 'message': f'Face registration successful with {len(all_encodings)} face samples. You can now log in with your face.'})
        except Exception as db_error:
            db.session.rollback()
            print(f"Database error during face registration: {db_error}") # From auth_app.py
            return jsonify({'success': False, 'message': 'Database error during registration. Please try again.'}), 500
        
    except Exception as e:
        print(f"Unexpected error during face registration: {e}") # From auth_app.py
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': 'An unexpected error occurred. Please try again.'}), 500


@app.route('/face_login', methods=['POST'])
def face_login():
    """
    Receives a single live face image for login.
    Compares it against all registered users' face encodings.
    """
    try:
        if not request.is_json:
            print("Error: Request is not JSON for face_login") # From auth_app.py
            return jsonify({'success': False, 'message': 'Invalid request format. JSON expected.'}), 400
            
        data = request.get_json()
        if not data:
            print("Error: No JSON data received for face_login") # From auth_app.py
            return jsonify({'success': False, 'message': 'No data received.'}), 400
            
        image_data = data.get('image')
        print("Received face login request.") # From auth_app.py

        if not image_data:
            print("Error: Face image required for login.") # From auth_app.py
            return jsonify({'success': False, 'message': 'Face image required for login.'}), 400

        live_encoding = get_face_encoding_from_image(image_data)
        if not live_encoding:
            print("No face detected in the live image for login.") # From auth_app.py
            return jsonify({'success': False, 'message': 'No face detected. Please position your face clearly in the camera and ensure good lighting.'}), 400

        users = User.query.filter(User.face_encodings_json.isnot(None)).all()
        print(f"Attempting to match against {len(users)} registered users with face data...") # From auth_app.py
        
        for user in users:
            if user.face_encodings_json:
                print(f"Comparing live encoding with stored encodings for user: {user.email}") # From auth_app.py
                if compare_face_encoding_to_stored(live_encoding, user.face_encodings_json):
                    session['user_id'] = user.id
                    print(f"Face login successful for user: {user.email}") # From auth_app.py
                    return jsonify({'success': True, 'message': f'Welcome back, {user.email}!'})
        
        print("Face not recognized against any registered user.") # From auth_app.py
        return jsonify({'success': False, 'message': 'Face not recognized. Please try again or use email/password login.'}), 401
        
    except Exception as e:
        print(f"Unexpected error during face login: {e}") # From auth_app.py
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': 'An error occurred during face login. Please try again.'}), 500

@app.route('/logout')
def logout():
    """Logs out the current user."""
    print(f"User {session.get('user_id')} logging out.") # From auth_app.py
    session.pop('user_id', None)
    flash("You have been logged out.", "info") # Added for consistency with other parts
    return redirect(url_for('auth_page'))

# Main application route (protected)
@app.route('/main_app')
@login_required
def main_app():
    """
    Renders the main image processing page only if the user is authenticated.
    """
    return render_template('index.html', 
                           caption=None, 
                           uploaded_image_url=None,
                           segmentation_image_url=None,
                           segmentation_metrics={})

# Predict route (protected)
@app.route('/predict', methods=['POST'])
@login_required
def predict():
    """
    Handles image upload, performs captioning and segmentation,
    and renders the results.
    """
    logger.info("Received request to /predict.")

    # Initialize variables for the template
    generated_caption = "N/A"
    uploaded_image_url = None
    segmentation_image_url = None
    segmentation_metrics = {} 

    if 'file' not in request.files:
        flash('No file part in the request.', 'error')
        logger.warning("No file part.")
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        flash('No selected file.', 'error')
        logger.warning("Empty filename.")
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        # Extract filename stem (without extension) for segmented image naming
        filename_stem, file_ext = os.path.splitext(filename)
        
        # Construct the full path to save the original image
        original_filepath = os.path.join(app.root_path, app.config['UPLOAD_FOLDER'], filename)
        file.save(original_filepath)
        uploaded_image_url = url_for('static', filename=f'uploads/{filename}')
        logger.info(f"Original image saved to: {original_filepath}")
        
        # --- Perform Image Captioning (using your provided correct method) ---
        try:
            logger.info(f"Starting caption generation for {original_filepath}...")
            # This is your original working call for captioning
            generated_caption = generate_caption_for_image(original_filepath)
            logger.info(f"Caption generated: '{generated_caption}'")
            flash("Image caption generated successfully!", 'success')
        except FileNotFoundError:
            flash(f"Error: Captioning model or vocabulary file not found for {original_filepath}. Check server logs.", 'error')
            logger.error(f"Captioning model/vocab not found during inference for {original_filepath}.")
            generated_caption = "Error: Captioning model/vocab not found."
        except RuntimeError as re:
            flash(f"Error: Captioning model not initialized or device issue. Check server logs.", 'error')
            logger.critical(f"Captioning model not initialized: {re}", exc_info=True)
            generated_caption = "Error: Captioning service unavailable."
        except Exception as e:
            flash(f"An unexpected error occurred during caption generation. Check server logs.", 'error')
            logger.critical(f"Error generating caption for {original_filepath}: {e}", exc_info=True)
            generated_caption = "Error: Could not generate caption."


        # --- Perform Image Segmentation ---
        if segmentation_model_yolo:
            logger.info(f"Starting segmentation for {original_filepath} using YOLO...")
            segmentation_image_url, segmentation_metrics = perform_segmentation(
                image_path=original_filepath,
                model_ref=segmentation_model_yolo,
                upload_folder=os.path.join(app.root_path, app.config['UPLOAD_FOLDER']),
                filename_stem=filename_stem
            )
            if 'error' in segmentation_metrics and segmentation_metrics['error']:
                 flash(f"Segmentation Error: {segmentation_metrics['error']}", 'error')
            elif segmentation_image_url:
                flash("Image segmentation performed successfully!", 'success')
            else:
                flash(f"Segmentation: {segmentation_metrics.get('status', 'No specific status.')}", 'info')
        else:
            flash("Segmentation model not initialized. Ensure 'ultralytics' is installed and model loaded.", 'error')
            logger.error("Segmentation model (YOLO) is not available.")
            segmentation_metrics['error'] = "Segmentation service unavailable."


        # Render the template with results for both tasks
        return render_template('index.html', 
                               caption=generated_caption, 
                               uploaded_image_url=uploaded_image_url,
                               segmentation_image_url=segmentation_image_url,
                               segmentation_metrics=segmentation_metrics)
    else:
        flash('Allowed image types are png, jpg, jpeg, gif.', 'error')
        logger.warning(f"Disallowed file type uploaded: {file.filename}")
        return redirect(request.url)

if __name__ == '__main__':
    logger.info("Starting Flask web application with integrated auth...")
    # This app will run on port 5000, handling both auth and image processing.
    app.run(debug=True, host='0.0.0.0', port=5000)
