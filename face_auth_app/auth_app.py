# # auth_app.py
# import os
# import base64
# import json
# import numpy as np
# import face_recognition # pip install face_recognition
# from flask import Flask, render_template, request, redirect, url_for, session, jsonify
# from flask_sqlalchemy import SQLAlchemy # pip install Flask-SQLAlchemy
# from werkzeug.security import generate_password_hash, check_password_hash # pip install Werkzeug
# from PIL import Image
# from io import BytesIO
# import logging

# # --- Setup ---
# app = Flask(__name__)
# app.config['SECRET_KEY'] = os.urandom(24) # Strong secret key for session management
# app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db' # SQLite database
# app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# db = SQLAlchemy(app)

# # Suppress some logging for cleaner output, but keep our custom prints
# logging.getLogger("werkzeug").setLevel(logging.ERROR)

# # --- Database Model ---
# class User(db.Model):
#     id = db.Column(db.Integer, primary_key=True)
#     email = db.Column(db.String(120), unique=True, nullable=False)
#     password_hash = db.Column(db.String(256), nullable=False)
#     # Store face encodings as JSON string of a list of floats (numpy arrays are not directly JSON serializable)
#     face_encodings_json = db.Column(db.Text, nullable=True) 

#     def __repr__(self):
#         return f'<User {self.email}>'

# # Create database tables if they don't exist
# with app.app_context():
#     db.create_all()
#     print("Database tables created/checked.")

# # --- Helper Functions for Facial Recognition ---
# def get_face_encoding_from_image(image_data_b64):
#     """
#     Decodes base64 image data, finds faces, and returns the first face's encoding.
#     Returns None if no face is found or on error.
#     """
#     try:
#         # Handle both formats: with and without data URL prefix
#         if ',' in image_data_b64:
#             image_bytes = base64.b64decode(image_data_b64.split(',')[1])
#         else:
#             image_bytes = base64.b64decode(image_data_b64)
            
#         img = Image.open(BytesIO(image_bytes)).convert('RGB')
        
#         # Convert PIL Image to numpy array (face_recognition expects numpy array)
#         img_np = np.array(img)
        
#         face_encodings = face_recognition.face_encodings(img_np)
#         if len(face_encodings) > 0:
#             print(f"Detected {len(face_encodings)} face(s) in image. Using the first one.")
#             return face_encodings[0].tolist() # Convert numpy array to list for JSON serialization
#         else:
#             print("No face detected in the provided image.")
#             return None # No face found
#     except Exception as e:
#         print(f"Error processing image for face encoding: {e}")
#         return None

# def compare_face_encoding_to_stored(live_encoding, stored_encodings_json):
#     """
#     Compares a live face encoding to a list of stored encodings for a user.
#     Returns True if a match is found, False otherwise.
#     """
#     if not live_encoding:
#         print("Live encoding is None, cannot compare.")
#         return False
#     if not stored_encodings_json:
#         print("Stored encodings JSON is None, cannot compare.")
#         return False
    
#     try:
#         # Convert JSON string back to a list of numpy arrays
#         stored_encodings = [np.array(e) for e in json.loads(stored_encodings_json)]
        
#         # Compare the live encoding against all stored encodings for this user
#         # tolerance: lower value means stricter match (0.6 is common default)
#         matches = face_recognition.compare_faces(stored_encodings, np.array(live_encoding), tolerance=0.6)
        
#         if True in matches:
#             print(f"Face match found! Matches: {matches}")
#         else:
#             print(f"No face match. Matches: {matches}")
        
#         return True in matches # True if any of the stored encodings match
#     except Exception as e:
#         print(f"Error comparing face encodings: {e}")
#         return False

# # --- Routes ---
# @app.route('/')
# def auth_page():
#     """Serves the authentication HTML page."""
#     if 'user_id' in session:
#         # If already logged in, redirect to a "dashboard" or directly to the main app
#         user = User.query.get(session['user_id'])
#         if user:
#             return f"""
#             <div style="text-align: center; margin-top: 100px; font-family: Arial, sans-serif;">
#                 <h1>Welcome, {user.email}!</h1>
#                 <p>You are successfully logged in.</p>
#                 <a href='/logout' style="color: #007bff; text-decoration: none;">Logout</a>
#             </div>
#             """
#         else:
#             session.pop('user_id', None) # Clear invalid session
#             return redirect(url_for('auth_page'))
#     return render_template('auth.html')

# @app.route('/register', methods=['POST'])
# def register():
#     """Handles traditional email/password registration."""
#     email = request.form['email']
#     password = request.form['password']
#     print(f"Received traditional registration request for: {email}")

#     if not email or not password:
#         print("Error: Email or password missing for traditional registration.")
#         return jsonify({'success': False, 'message': 'Email and password are required.'}), 400

#     existing_user = User.query.filter_by(email=email).first()
#     if existing_user:
#         print(f"Error: Email {email} already registered.")
#         return jsonify({'success': False, 'message': 'Email already registered.'}), 409

#     hashed_password = generate_password_hash(password)
#     new_user = User(email=email, password_hash=hashed_password)
    
#     try:
#         db.session.add(new_user)
#         db.session.commit()
#         print(f"Traditional registration successful for: {email}")
#         return jsonify({'success': True, 'message': 'Registration successful. You can now log in.'})
#     except Exception as e:
#         db.session.rollback()
#         print(f"Database error during traditional registration: {e}")
#         return jsonify({'success': False, 'message': 'Database error during registration.'}), 500


# @app.route('/login', methods=['POST'])
# def login():
#     """Handles traditional email/password login."""
#     email = request.form['email']
#     password = request.form['password']
#     print(f"Received traditional login request for: {email}")

#     user = User.query.filter_by(email=email).first()

#     if user and check_password_hash(user.password_hash, password):
#         session['user_id'] = user.id
#         print(f"Traditional login successful for: {email}")
#         return jsonify({'success': True, 'message': 'Login successful.'})
#     else:
#         print(f"Traditional login failed for: {email}")
#         return jsonify({'success': False, 'message': 'Invalid email or password.'}), 401

# @app.route('/face_register', methods=['POST'])
# def face_register():
#     """
#     Receives face images for registration.
#     Expects email, password, and a list of base64 image data from the frontend.
#     """
#     try:
#         data = request.get_json()
#         email = data.get('email')
#         password = data.get('password')
#         image_data_list = data.get('images', [])
        
#         print(f"Received face registration request for: {email} with {len(image_data_list)} images.")

#         if not email or not password or not image_data_list:
#             print("Error: Missing email, password or image data for face registration.")
#             return jsonify({'success': False, 'message': 'Email, password, and face images are required.'}), 400

#         existing_user = User.query.filter_by(email=email).first()
#         if existing_user:
#             print(f"Error: Email {email} already registered for face registration.")
#             return jsonify({'success': False, 'message': 'Email already registered.'}), 409

#         all_encodings = []
#         for i, img_b64 in enumerate(image_data_list):
#             print(f"Processing image {i+1}/{len(image_data_list)} for face encoding...")
#             encoding = get_face_encoding_from_image(img_b64)
#             if encoding:
#                 all_encodings.append(encoding)
#             else:
#                 print(f"No face detected in image {i+1}. This image will be skipped.")
        
#         if not all_encodings:
#             print("Error: No detectable faces found in any of the provided images for face registration.")
#             return jsonify({'success': False, 'message': 'No detectable faces in the provided images. Please try again with clearer images.'}), 400

#         hashed_password = generate_password_hash(password)
#         # Store multiple encodings as JSON string
#         face_encodings_json = json.dumps(all_encodings) 
        
#         new_user = User(email=email, password_hash=hashed_password, face_encodings_json=face_encodings_json)
        
#         db.session.add(new_user)
#         db.session.commit()
#         print(f"Face registration successful for: {email}. Stored {len(all_encodings)} encodings.")
#         return jsonify({'success': True, 'message': f'Face registration successful with {len(all_encodings)} face samples. You can now log in with your face.'})
        
#     except Exception as e:
#         db.session.rollback()
#         print(f"Error during face registration: {e}")
#         return jsonify({'success': False, 'message': 'Error during face registration. Please try again.'}), 500


# @app.route('/face_login', methods=['POST'])
# def face_login():
#     """
#     Receives a single live face image for login.
#     Compares it against all registered users' face encodings.
#     """
#     try:
#         data = request.get_json()
#         image_data = data.get('image')
#         print("Received face login request.")

#         if not image_data:
#             print("Error: Face image required for login.")
#             return jsonify({'success': False, 'message': 'Face image required for login.'}), 400

#         live_encoding = get_face_encoding_from_image(image_data)
#         if not live_encoding:
#             print("No face detected in the live image for login.")
#             return jsonify({'success': False, 'message': 'No face detected. Please position your face clearly in the camera.'}), 400

#         users = User.query.filter(User.face_encodings_json.isnot(None)).all()
#         print(f"Attempting to match against {len(users)} registered users with face data...")
        
#         for user in users:
#             if user.face_encodings_json:
#                 print(f"Comparing live encoding with stored encodings for user: {user.email}")
#                 if compare_face_encoding_to_stored(live_encoding, user.face_encodings_json):
#                     session['user_id'] = user.id
#                     print(f"Face login successful for user: {user.email}")
#                     return jsonify({'success': True, 'message': f'Welcome back, {user.email}!'})
        
#         print("Face not recognized against any registered user.")
#         return jsonify({'success': False, 'message': 'Face not recognized. Please try again or use email/password login.'}), 401
        
#     except Exception as e:
#         print(f"Error during face login: {e}")
#         return jsonify({'success': False, 'message': 'Error during face login. Please try again.'}), 500

# @app.route('/logout')
# def logout():
#     """Logs out the current user."""
#     print(f"User {session.get('user_id')} logging out.")
#     session.pop('user_id', None)
#     return redirect(url_for('auth_page'))

# # --- Run App ---
# if __name__ == '__main__':
#     print("Starting Authentication Server...")
#     print("Database will be created/updated at: users.db")
#     print("Access the authentication page at: http://127.0.0.1:5001")
#     app.run(debug=True, port=5001)









# auth_app.py
import os
import base64
import json
import numpy as np
import face_recognition # pip install face_recognition
from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from flask_sqlalchemy import SQLAlchemy # pip install Flask-SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash # pip install Werkzeug
from PIL import Image
from io import BytesIO
import logging

# --- Setup ---
app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24) # Strong secret key for session management
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db' # SQLite database
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# Suppress some logging for cleaner output, but keep our custom prints
logging.getLogger("werkzeug").setLevel(logging.ERROR)

# --- Database Model ---
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
    print("Database tables created/checked.")

# --- Helper Functions for Facial Recognition ---
def get_face_encoding_from_image(image_data_b64):
    """
    Decodes base64 image data, finds faces, and returns the first face's encoding.
    Returns None if no face is found or on error.
    """
    try:
        print(f"Processing image data of length: {len(image_data_b64)}")
        
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
            print(f"Base64 decode error: {decode_error}")
            return None
            
        print(f"Decoded image bytes length: {len(image_bytes)}")
        
        # Open and convert image
        try:
            img = Image.open(BytesIO(image_bytes))
            print(f"Image opened successfully. Format: {img.format}, Size: {img.size}, Mode: {img.mode}")
            
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
                print(f"Converted image to RGB mode")
            
        except Exception as img_error:
            print(f"Image opening/conversion error: {img_error}")
            return None
        
        # Convert PIL Image to numpy array (face_recognition expects numpy array)
        img_np = np.array(img)
        print(f"Numpy array shape: {img_np.shape}")
        
        # Find face encodings
        try:
            face_locations = face_recognition.face_locations(img_np)
            print(f"Found {len(face_locations)} face location(s)")
            
            if len(face_locations) == 0:
                print("No faces detected in the image")
                return None
            
            face_encodings = face_recognition.face_encodings(img_np, face_locations)
            print(f"Generated {len(face_encodings)} face encoding(s)")
            
            if len(face_encodings) > 0:
                encoding = face_encodings[0]
                print(f"Face encoding shape: {encoding.shape}")
                return encoding.tolist() # Convert numpy array to list for JSON serialization
            else:
                print("No face encodings generated despite face locations found")
                return None
                
        except Exception as face_error:
            print(f"Face recognition processing error: {face_error}")
            return None
            
    except Exception as e:
        print(f"General error processing image for face encoding: {e}")
        return None

def compare_face_encoding_to_stored(live_encoding, stored_encodings_json):
    """
    Compares a live face encoding to a list of stored encodings for a user.
    Returns True if a match is found, False otherwise.
    """
    if not live_encoding:
        print("Live encoding is None, cannot compare.")
        return False
    if not stored_encodings_json:
        print("Stored encodings JSON is None, cannot compare.")
        return False
    
    try:
        # Convert JSON string back to a list of numpy arrays
        stored_encodings_list = json.loads(stored_encodings_json)
        if not stored_encodings_list:
            print("No stored encodings found in JSON")
            return False
            
        stored_encodings = [np.array(e) for e in stored_encodings_list]
        print(f"Comparing against {len(stored_encodings)} stored encodings")
        
        # Compare the live encoding against all stored encodings for this user
        # tolerance: lower value means stricter match (0.6 is common default)
        matches = face_recognition.compare_faces(stored_encodings, np.array(live_encoding), tolerance=0.6)
        
        match_found = True in matches
        print(f"Face comparison result: {match_found}. Matches: {matches}")
        
        return match_found
    except Exception as e:
        print(f"Error comparing face encodings: {e}")
        return False

# --- Routes ---
@app.route('/')
def auth_page():
    """Serves the authentication HTML page."""
    if 'user_id' in session:
        # If already logged in, redirect to a "dashboard" or directly to the main app
        user = User.query.get(session['user_id'])
        if user:
            return f"""
            <div style="text-align: center; margin-top: 100px; font-family: Arial, sans-serif;">
                <h1>Welcome, {user.email}!</h1>
                <p>You are successfully logged in.</p>
                <a href='/logout' style="color: #007bff; text-decoration: none;">Logout</a>
            </div>
            """
        else:
            session.pop('user_id', None) # Clear invalid session
            return redirect(url_for('auth_page'))
    return render_template('auth.html')

@app.route('/register', methods=['POST'])
def register():
    """Handles traditional email/password registration."""
    email = request.form['email']
    password = request.form['password']
    print(f"Received traditional registration request for: {email}")

    if not email or not password:
        print("Error: Email or password missing for traditional registration.")
        return jsonify({'success': False, 'message': 'Email and password are required.'}), 400

    existing_user = User.query.filter_by(email=email).first()
    if existing_user:
        print(f"Error: Email {email} already registered.")
        return jsonify({'success': False, 'message': 'Email already registered.'}), 409

    hashed_password = generate_password_hash(password)
    new_user = User(email=email, password_hash=hashed_password)
    
    try:
        db.session.add(new_user)
        db.session.commit()
        print(f"Traditional registration successful for: {email}")
        return jsonify({'success': True, 'message': 'Registration successful. You can now log in.'})
    except Exception as e:
        db.session.rollback()
        print(f"Database error during traditional registration: {e}")
        return jsonify({'success': False, 'message': 'Database error during registration.'}), 500


@app.route('/login', methods=['POST'])
def login():
    """Handles traditional email/password login."""
    email = request.form['email']
    password = request.form['password']
    print(f"Received traditional login request for: {email}")

    user = User.query.filter_by(email=email).first()

    if user and check_password_hash(user.password_hash, password):
        session['user_id'] = user.id
        print(f"Traditional login successful for: {email}")
        return jsonify({'success': True, 'message': 'Login successful.'})
    else:
        print(f"Traditional login failed for: {email}")
        return jsonify({'success': False, 'message': 'Invalid email or password.'}), 401

@app.route('/face_register', methods=['POST'])
def face_register():
    """
    Receives face images for registration.
    Expects email, password, and a list of base64 image data from the frontend.
    """
    try:
        # Validate content type
        if not request.is_json:
            print("Error: Request is not JSON")
            return jsonify({'success': False, 'message': 'Invalid request format. JSON expected.'}), 400
            
        data = request.get_json()
        if not data:
            print("Error: No JSON data received")
            return jsonify({'success': False, 'message': 'No data received.'}), 400
            
        email = data.get('email', '').strip()
        password = data.get('password', '')
        image_data_list = data.get('images', [])
        
        print(f"Received face registration request for: {email} with {len(image_data_list)} images.")

        # Validate input data
        if not email or not password or not image_data_list:
            print("Error: Missing email, password or image data for face registration.")
            return jsonify({'success': False, 'message': 'Email, password, and face images are required.'}), 400

        # Validate email format
        import re
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(email_pattern, email):
            print(f"Error: Invalid email format: {email}")
            return jsonify({'success': False, 'message': 'Please enter a valid email address.'}), 400

        # Validate password length
        if len(password) < 6:
            print("Error: Password too short")
            return jsonify({'success': False, 'message': 'Password must be at least 6 characters long.'}), 400

        # Check if user already exists
        existing_user = User.query.filter_by(email=email).first()
        if existing_user:
            print(f"Error: Email {email} already registered for face registration.")
            return jsonify({'success': False, 'message': 'Email already registered.'}), 409

        # Process face images
        all_encodings = []
        for i, img_b64 in enumerate(image_data_list):
            print(f"Processing image {i+1}/{len(image_data_list)} for face encoding...")
            
            if not img_b64:
                print(f"Warning: Image {i+1} is empty, skipping")
                continue
                
            encoding = get_face_encoding_from_image(img_b64)
            if encoding:
                all_encodings.append(encoding)
                print(f"Successfully processed image {i+1}")
            else:
                print(f"Failed to process image {i+1} - no face detected or processing error")
        
        if not all_encodings:
            print("Error: No detectable faces found in any of the provided images for face registration.")
            return jsonify({'success': False, 'message': 'No detectable faces in the provided images. Please try again with clearer images showing your face clearly.'}), 400

        # Create new user
        hashed_password = generate_password_hash(password)
        face_encodings_json = json.dumps(all_encodings) 
        
        new_user = User(email=email, password_hash=hashed_password, face_encodings_json=face_encodings_json)
        
        try:
            db.session.add(new_user)
            db.session.commit()
            print(f"Face registration successful for: {email}. Stored {len(all_encodings)} encodings.")
            return jsonify({'success': True, 'message': f'Face registration successful with {len(all_encodings)} face samples. You can now log in with your face.'})
        except Exception as db_error:
            db.session.rollback()
            print(f"Database error during face registration: {db_error}")
            return jsonify({'success': False, 'message': 'Database error during registration. Please try again.'}), 500
        
    except Exception as e:
        print(f"Unexpected error during face registration: {e}")
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
        # Validate content type
        if not request.is_json:
            print("Error: Request is not JSON")
            return jsonify({'success': False, 'message': 'Invalid request format. JSON expected.'}), 400
            
        data = request.get_json()
        if not data:
            print("Error: No JSON data received")
            return jsonify({'success': False, 'message': 'No data received.'}), 400
            
        image_data = data.get('image')
        print("Received face login request.")

        if not image_data:
            print("Error: Face image required for login.")
            return jsonify({'success': False, 'message': 'Face image required for login.'}), 400

        # Process the live image
        live_encoding = get_face_encoding_from_image(image_data)
        if not live_encoding:
            print("No face detected in the live image for login.")
            return jsonify({'success': False, 'message': 'No face detected. Please position your face clearly in the camera and ensure good lighting.'}), 400

        # Get all users with face data
        users = User.query.filter(User.face_encodings_json.isnot(None)).all()
        print(f"Attempting to match against {len(users)} registered users with face data...")
        
        for user in users:
            if user.face_encodings_json:
                print(f"Comparing live encoding with stored encodings for user: {user.email}")
                if compare_face_encoding_to_stored(live_encoding, user.face_encodings_json):
                    session['user_id'] = user.id
                    print(f"Face login successful for user: {user.email}")
                    return jsonify({'success': True, 'message': f'Welcome back, {user.email}!'})
        
        print("Face not recognized against any registered user.")
        return jsonify({'success': False, 'message': 'Face not recognized. Please try again or use email/password login.'}), 401
        
    except Exception as e:
        print(f"Unexpected error during face login: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': 'An error occurred during face login. Please try again.'}), 500

@app.route('/logout')
def logout():
    """Logs out the current user."""
    print(f"User {session.get('user_id')} logging out.")
    session.pop('user_id', None)
    return redirect(url_for('auth_page'))

# --- Run App ---
if __name__ == '__main__':
    print("Starting Authentication Server...")
    print("Database will be created/updated at: users.db")
    print("Access the authentication page at: http://127.0.0.1:5001")
    app.run(debug=True, port=5001)

