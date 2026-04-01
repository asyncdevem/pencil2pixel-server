from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image, ImageEnhance, ImageFilter
import io
import base64
import os
import numpy as np
import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '0'
import cv2
from datetime import datetime, timedelta
from functools import wraps
import jwt
import psycopg2
from psycopg2.extras import RealDictCursor
from werkzeug.security import generate_password_hash, check_password_hash
import uuid

# Apply GFPGAN compatibility patch BEFORE importing GFPGAN
import sys
import torchvision.transforms.functional as TF

# Create the missing module that basicsr expects
class FunctionalTensorModule:
    """Compatibility shim for torchvision.transforms.functional_tensor"""
    
    @staticmethod
    def rgb_to_grayscale(img, num_output_channels=1):
        return TF.rgb_to_grayscale(img, num_output_channels)

# Inject the compatibility module
sys.modules['torchvision.transforms.functional_tensor'] = FunctionalTensorModule()
print("✓ Applied GFPGAN compatibility patch for torchvision")

app = Flask(__name__)
CORS(app)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'your-secret-key-change-in-production')
app.config['DATABASE_URL'] = os.environ.get('DATABASE_URL', 'postgresql://neondb_owner:npg_YjbBoyQ7gI8M@ep-red-butterfly-amb74pm5-pooler.c-5.us-east-1.aws.neon.tech/neondb%ssslmode=require')

# Quality enhancement settings
QUALITY_PRESETS = {
    'low': {'resolution': 512, 'sharpen': 1.0, 'enhance': 1.0},
    'medium': {'resolution': 512, 'sharpen': 1.3, 'enhance': 1.1},
    'high': {'resolution': 512, 'sharpen': 1.5, 'enhance': 1.2},
    'ultra': {'resolution': 512, 'sharpen': 1.8, 'enhance': 1.3}
}

# Default attributes for 4-channel model (4 tone controls: skin, hair, eye, lip)
DEFAULT_ATTRIBUTES = [0.25, 0.75, 0.33, 0.33]

# --- MODEL ARCHITECTURE ---
class SketchToImageGenerator(nn.Module):
    def __init__(self, attr_dim=4):
        super().__init__()
        
        def down(i, o, n=True):
            l = [nn.Conv2d(i, o, 4, 2, 1, bias=False)]
            if n: l.append(nn.InstanceNorm2d(o))
            l.append(nn.LeakyReLU(0.2, True))
            return nn.Sequential(*l)
        
        def up(i, o, dr=0.0):
            l = [nn.ConvTranspose2d(i, o, 4, 2, 1, bias=False), nn.InstanceNorm2d(o), nn.ReLU(True)]
            if dr: l.append(nn.Dropout(dr))
            return nn.Sequential(*l)
        
        # Configured for 4-channels (RGB Sketch + 1-ch Mask)
        self.d1 = down(4, 64, False)
        self.d2, self.d3, self.d4 = down(64, 128), down(128, 256), down(256, 512)
        self.d5, self.d6, self.d7, self.d8 = down(512, 512), down(512, 512), down(512, 512), down(512, 512, False)
        
        self.up1 = up(512 + attr_dim, 512, 0.5)
        self.up2, self.up3, self.up4 = up(1024, 512, 0.5), up(1024, 512, 0.5), up(1024, 512)
        self.up5, self.up6, self.up7 = up(1024, 256), up(512, 128), up(256, 64)
        
        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128, 3, 4, padding=1),
            nn.Tanh()
        )
    
    def forward(self, s, m, a):
        x = torch.cat([s, m], dim=1)  # 4-Channel Fusion
        d1 = self.d1(x); d2 = self.d2(d1); d3 = self.d3(d2); d4 = self.d4(d3)
        d5 = self.d5(d4); d6 = self.d6(d5); d7 = self.d7(d6); d8 = self.d8(d7)
        
        a_exp = a.view(a.size(0), -1, 1, 1).expand(-1, -1, d8.size(2), d8.size(3))
        u1 = self.up1(torch.cat([d8, a_exp], 1))
        u2 = self.up2(torch.cat([u1, d7], 1)); u3 = self.up3(torch.cat([u2, d6], 1))
        u4 = self.up4(torch.cat([u3, d5], 1)); u5 = self.up5(torch.cat([u4, d4], 1))
        u6 = self.up6(torch.cat([u5, d3], 1)); u7 = self.up7(torch.cat([u6, d2], 1))
        
        return self.final(torch.cat([u7, d1], 1))

# --- LOAD MODEL ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# First, check the model checkpoint to determine attr_dim
checkpoint_path = 'model/pencil2pixel.pth'
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Determine attr_dim from the checkpoint
    # up1 is ConvTranspose2d with weight shape [in_channels, out_channels, kernel_h, kernel_w]
    # In forward pass: up1(torch.cat([d8, a_exp], 1)) where d8 has 512 channels
    # So: up1 in_channels = 512 + attr_dim
    if 'up1.0.weight' in checkpoint:
        up1_in_channels = checkpoint['up1.0.weight'].shape[0]  # First dim is in_channels for ConvTranspose2d
        d8_channels = 512
        attr_dim = up1_in_channels - d8_channels
        print(f"Detected attr_dim from checkpoint: {attr_dim}")
    else:
        attr_dim = 4  # Default fallback
        print(f"Could not detect attr_dim, using default: {attr_dim}")
else:
    attr_dim = 4
    print(f"Checkpoint not found, using default attr_dim: {attr_dim}")

# Create model with correct attr_dim
model = SketchToImageGenerator(attr_dim=attr_dim).to(device)

# Load checkpoint with 4-channel adaptation
if os.path.exists(checkpoint_path):
    # Surgery: Adapting 3-channel weights to 4-channel model if needed
    if 'd1.0.weight' in checkpoint and checkpoint['d1.0.weight'].shape[1] == 3:
        old_weight = checkpoint['d1.0.weight']
        new_channel = torch.mean(old_weight, dim=1, keepdim=True)
        checkpoint['d1.0.weight'] = torch.cat([old_weight, new_channel], dim=1)
    
    model.load_state_dict(checkpoint, strict=False)
    print(f"Model loaded successfully with attr_dim={attr_dim}")
else:
    print(f"Warning: Model file not found at {checkpoint_path}")

model.eval()

transform = T.Compose([
    T.Resize((512, 512)),
    T.ToTensor(),
    T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# --- LOAD GFPGAN ENHANCER ---
gfpgan_available = False
gfpgan_restorer = None

try:
    from gfpgan import GFPGANer
    
    # Initialize GFPGAN
    gfpgan_restorer = GFPGANer(
        model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth',
        upscale=1,
        arch='clean',
        channel_multiplier=2,
        bg_upsampler=None,
        device=device
    )
    print("✓ GFPGAN enhancer loaded successfully")
    gfpgan_available = True
except ImportError as e:
    print(f"⚠ GFPGAN import error: {e}")
    print("  GFPGAN enhancement disabled - using PIL enhancement only")
except Exception as e:
    print(f"⚠ GFPGAN initialization error: {e}")
    print("  GFPGAN enhancement disabled - using PIL enhancement only")

# --- IMAGE ENHANCEMENT FUNCTIONS ---
def enhance_output_image(img, quality='medium', use_gfpgan=True):
    """
    Apply post-processing enhancements to improve output quality
    If GFPGAN is available and use_gfpgan=True, use GFPGAN, otherwise use PIL
    """
    if use_gfpgan and gfpgan_available:
        # Use GFPGAN enhancement
        try:
            # Convert PIL to numpy array in BGR format for GFPGAN
            img_np = np.array(img)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            
            # Apply GFPGAN
            _, _, restored_img = gfpgan_restorer.enhance(
                img_bgr, 
                has_aligned=False, 
                only_center_face=False, 
                paste_back=True
            )
            
            # Convert back to PIL RGB
            restored_rgb = cv2.cvtColor(restored_img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(restored_rgb)
            
            return img
        except Exception as e:
            print(f"GFPGAN enhancement failed: {e}, falling back to PIL")
            # Fall through to PIL enhancement
    
    # PIL-based enhancement (fallback or when GFPGAN disabled)
    preset = QUALITY_PRESETS.get(quality, QUALITY_PRESETS['medium'])
    
    # 1. Sharpen the image
    if preset['sharpen'] > 1.0:
        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(preset['sharpen'])
    
    # 2. Enhance contrast
    if preset['enhance'] > 1.0:
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(preset['enhance'])
    
    # 3. Slight color enhancement
    enhancer = ImageEnhance.Color(img)
    img = enhancer.enhance(1.1)
    
    # 4. Apply unsharp mask for better details
    img = img.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
    
    return img

def preprocess_input_image(img):
    """
    Preprocess input image to reduce artifacts and improve quality
    """
    # Convert to RGB if needed
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Apply slight denoising
    img = img.filter(ImageFilter.MedianFilter(size=3))
    
    # Enhance contrast slightly
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.05)
    
    return img

def create_mask_from_sketch(img, threshold=250):
    """
    Create binary mask from sketch image
    White areas (background) = 0, Dark areas (sketch) = 1
    """
    gray_np = np.array(img.convert('L').resize((512, 512)))
    mask_np = np.where(gray_np < threshold, 1.0, 0.0)
    return mask_np

# --- DATABASE SETUP ---
def init_db():
    conn = psycopg2.connect(app.config['DATABASE_URL'])
    c = conn.cursor()
    
    # Users table
    c.execute('''CREATE TABLE IF NOT EXISTS users (
        id SERIAL PRIMARY KEY,
        user_id TEXT UNIQUE NOT NULL,
        username TEXT UNIQUE NOT NULL,
        email TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )''')
    
    # Image history table
    c.execute('''CREATE TABLE IF NOT EXISTS image_history (
        id SERIAL PRIMARY KEY,
        image_id TEXT UNIQUE NOT NULL,
        user_id TEXT NOT NULL,
        original_filename TEXT,
        generated_image BYTEA NOT NULL,
        attributes TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users (user_id) ON DELETE CASCADE
    )''')
    
    conn.commit()
    conn.close()

def get_db():
    conn = psycopg2.connect(app.config['DATABASE_URL'], cursor_factory=RealDictCursor)
    return conn

# Initialize database on startup
init_db()

# --- AUTHENTICATION MIDDLEWARE ---
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization')
        
        if not token:
            return jsonify({'error': 'Token is missing'}), 401
        
        try:
            if token.startswith('Bearer '):
                token = token[7:]
            data = jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
            current_user_id = data['user_id']
        except jwt.ExpiredSignatureError:
            return jsonify({'error': 'Token has expired'}), 401
        except jwt.InvalidTokenError:
            return jsonify({'error': 'Invalid token'}), 401
        
        return f(current_user_id, *args, **kwargs)
    
    return decorated

# --- API ENDPOINTS ---

# --- AUTHENTICATION ENDPOINTS ---
@app.route('/auth/signup', methods=['POST'])
def signup():
    try:
        data = request.get_json()
        
        if not data or not data.get('username') or not data.get('email') or not data.get('password'):
            return jsonify({'error': 'Username, email, and password are required'}), 400
        
        username = data['username']
        email = data['email']
        password = data['password']
        
        # Validate password length
        if len(password) < 6:
            return jsonify({'error': 'Password must be at least 6 characters'}), 400
        
        conn = get_db()
        c = conn.cursor()
        
        # Check if user already exists
        c.execute('SELECT * FROM users WHERE username = %s OR email = %s', (username, email))
        if c.fetchone():
            conn.close()
            return jsonify({'error': 'Username or email already exists'}), 409
        
        # Create new user
        user_id = str(uuid.uuid4())
        password_hash = generate_password_hash(password)
        
        c.execute('INSERT INTO users (user_id, username, email, password_hash) VALUES (%s, %s, %s, %s)',
                  (user_id, username, email, password_hash))
        conn.commit()
        conn.close()
        
        # Generate token
        token = jwt.encode({
            'user_id': user_id,
            'exp': datetime.utcnow() + timedelta(days=7)
        }, app.config['SECRET_KEY'], algorithm='HS256')
        
        return jsonify({
            'message': 'User created successfully',
            'user': {
                'user_id': user_id,
                'username': username,
                'email': email
            },
            'token': token
        }), 201
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/auth/login', methods=['POST'])
def login():
    try:
        data = request.get_json()
        
        if not data or not data.get('email') or not data.get('password'):
            return jsonify({'error': 'Email and password are required'}), 400
        
        email = data['email']
        password = data['password']
        
        conn = get_db()
        c = conn.cursor()
        
        c.execute('SELECT * FROM users WHERE email = %s', (email,))
        user = c.fetchone()
        conn.close()
        
        if not user or not check_password_hash(user['password_hash'], password):
            return jsonify({'error': 'Invalid email or password'}), 401
        
        # Generate token
        token = jwt.encode({
            'user_id': user['user_id'],
            'exp': datetime.utcnow() + timedelta(days=7)
        }, app.config['SECRET_KEY'], algorithm='HS256')
        
        return jsonify({
            'message': 'Login successful',
            'user': {
                'user_id': user['user_id'],
                'username': user['username'],
                'email': user['email']
            },
            'token': token
        }), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/auth/profile', methods=['GET'])
@token_required
def get_profile(current_user_id):
    try:
        conn = get_db()
        c = conn.cursor()
        
        c.execute('SELECT user_id, username, email, created_at FROM users WHERE user_id = %s', (current_user_id,))
        user = c.fetchone()
        
        # Get image count
        c.execute('SELECT COUNT(*) as count FROM image_history WHERE user_id = %s', (current_user_id,))
        image_count = c.fetchone()['count']
        
        conn.close()
        
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        return jsonify({
            'user': {
                'user_id': user['user_id'],
                'username': user['username'],
                'email': user['email'],
                'created_at': user['created_at'],
                'total_images': image_count
            }
        }), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/auth/profile', methods=['PUT'])
@token_required
def update_profile(current_user_id):
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        conn = get_db()
        c = conn.cursor()
        
        updates = []
        params = []
        
        if 'username' in data:
            # Check if username already exists
            c.execute('SELECT * FROM users WHERE username = %s AND user_id != %s', (data['username'], current_user_id))
            if c.fetchone():
                conn.close()
                return jsonify({'error': 'Username already exists'}), 409
            updates.append('username = %s')
            params.append(data['username'])
        
        if 'email' in data:
            # Check if email already exists
            c.execute('SELECT * FROM users WHERE email = %s AND user_id != %s', (data['email'], current_user_id))
            if c.fetchone():
                conn.close()
                return jsonify({'error': 'Email already exists'}), 409
            updates.append('email = %s')
            params.append(data['email'])
        
        if 'password' in data:
            if len(data['password']) < 6:
                conn.close()
                return jsonify({'error': 'Password must be at least 6 characters'}), 400
            updates.append('password_hash = %s')
            params.append(generate_password_hash(data['password']))
        
        if not updates:
            conn.close()
            return jsonify({'error': 'No valid fields to update'}), 400
        
        updates.append('updated_at = CURRENT_TIMESTAMP')
        params.append(current_user_id)
        
        query = f"UPDATE users SET {', '.join(updates)} WHERE user_id = %s"
        c.execute(query, params)
        conn.commit()
        
        c.execute('SELECT user_id, username, email, updated_at FROM users WHERE user_id = %s', (current_user_id,))
        user = c.fetchone()
        conn.close()
        
        return jsonify({
            'message': 'Profile updated successfully',
            'user': {
                'user_id': user['user_id'],
                'username': user['username'],
                'email': user['email'],
                'updated_at': user['updated_at']
            }
        }), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# --- IMAGE HISTORY ENDPOINTS ---
@app.route('/history', methods=['GET'])
@token_required
def get_history(current_user_id):
    try:
        page = request.args.get('page', 1, type=int)
        limit = request.args.get('limit', 10, type=int)
        offset = (page - 1) * limit
        
        conn = get_db()
        c = conn.cursor()
        
        c.execute('''SELECT image_id, original_filename, attributes, created_at 
                     FROM image_history 
                     WHERE user_id = %s 
                     ORDER BY created_at DESC 
                     LIMIT %s OFFSET %s''', (current_user_id, limit, offset))
        images = c.fetchall()
        
        c.execute('SELECT COUNT(*) as total FROM image_history WHERE user_id = %s', (current_user_id,))
        total = c.fetchone()['total']
        
        conn.close()
        
        return jsonify({
            'images': [{
                'image_id': img['image_id'],
                'original_filename': img['original_filename'],
                'attributes': img['attributes'],
                'created_at': img['created_at']
            } for img in images],
            'pagination': {
                'page': page,
                'limit': limit,
                'total': total,
                'pages': (total + limit - 1) // limit
            }
        }), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/history/<image_id>', methods=['GET'])
@token_required
def get_history_image(current_user_id, image_id):
    try:
        conn = get_db()
        c = conn.cursor()
        
        c.execute('SELECT * FROM image_history WHERE image_id = %s AND user_id = %s', (image_id, current_user_id))
        image = c.fetchone()
        conn.close()
        
        if not image:
            return jsonify({'error': 'Image not found'}), 404
        
        return send_file(
            io.BytesIO(image['generated_image']),
            mimetype='image/png',
            as_attachment=True,
            download_name=f"{image['original_filename'] or 'generated'}.png"
        )
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/history/<image_id>', methods=['DELETE'])
@token_required
def delete_history_image(current_user_id, image_id):
    try:
        conn = get_db()
        c = conn.cursor()
        
        c.execute('DELETE FROM image_history WHERE image_id = %s AND user_id = %s', (image_id, current_user_id))
        
        if c.rowcount == 0:
            conn.close()
            return jsonify({'error': 'Image not found'}), 404
        
        conn.commit()
        conn.close()
        
        return jsonify({'message': 'Image deleted successfully'}), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/history', methods=['DELETE'])
@token_required
def clear_history(current_user_id):
    try:
        conn = get_db()
        c = conn.cursor()
        
        c.execute('DELETE FROM image_history WHERE user_id = %s', (current_user_id,))
        deleted_count = c.rowcount
        
        conn.commit()
        conn.close()
        
        return jsonify({
            'message': 'History cleared successfully',
            'deleted_count': deleted_count
        }), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'service': 'pencil2pixel-api',
        'device': device,
        'gfpgan_available': gfpgan_available,
        'model_loaded': os.path.exists('model/pencil2pixel.pth')
    }), 200

@app.route('/', methods=['GET'])
def index():
    return jsonify({
        'message': 'Pencil2Pixel API is running',
        'status': 'ok',
        'endpoints': {
            'health': '/health',
            'auth': {
                'signup': '/auth/signup',
                'login': '/auth/login',
                'profile': '/auth/profile'
            },
            'generate': {
                'single': '/generate',
                'batch': '/generate-batch'
            },
            'history': '/history'
        }
    }), 200

@app.route('/generate', methods=['POST'])
@token_required
def generate(current_user_id):
    try:
        # Check if image file is present
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'Empty filename'}), 400
        
        # Get quality setting
        quality = request.form.get('quality', 'medium')
        if quality not in QUALITY_PRESETS:
            quality = 'medium'
        
        # Get enhancement method (gfpgan or pil)
        enhancement = request.form.get('enhancement', 'gfpgan')  # 'gfpgan' or 'pil'
        use_gfpgan = enhancement == 'gfpgan' and gfpgan_available
        
        # Get attributes (optional - can be None)
        attributes = request.form.get('attributes', None)
        if attributes and attr_dim > 0:
            attr_list = [float(x) for x in attributes.split(',')]
            if len(attr_list) != attr_dim:
                return jsonify({'error': f'Attributes must be {attr_dim} values'}), 400
            attr_tensor = torch.tensor([attr_list], dtype=torch.float32).to(device)
        else:
            # No attributes or model doesn't use them - use zeros
            attr_tensor = torch.zeros(1, max(attr_dim, 1), dtype=torch.float32).to(device)
        
        # Process image with preprocessing
        img = Image.open(file.stream).convert('RGB')
        original_size = img.size
        
        # Preprocess input
        img = preprocess_input_image(img)
        
        # Create sketch tensor
        sketch_tensor = transform(img).unsqueeze(0).to(device)
        
        # Create mask tensor (auto-masking)
        mask_np = create_mask_from_sketch(img)
        mask_tensor = torch.from_numpy(mask_np).float().unsqueeze(0).unsqueeze(0).to(device)
        
        # Generate
        with torch.no_grad():
            output = model(sketch_tensor, mask_tensor, attr_tensor)
        
        # Denormalize and convert to PIL
        output = (output.squeeze().cpu() + 1) / 2
        output_img = T.ToPILImage()(output)
        
        # Apply post-processing enhancements
        output_img = enhance_output_image(output_img, quality, use_gfpgan=use_gfpgan)
        
        # Optionally resize to larger size for better quality
        upscale = request.form.get('upscale', 'false').lower() == 'true'
        if upscale:
            # Upscale to 2x using high-quality resampling
            new_size = (1024, 1024)
            output_img = output_img.resize(new_size, Image.Resampling.LANCZOS)
        
        # Save to history if requested
        save_to_history = request.form.get('save', 'false').lower() == 'true'
        image_id = None
        
        if save_to_history:
            img_io = io.BytesIO()
            output_img.save(img_io, 'PNG')
            img_bytes = img_io.getvalue()
            
            image_id = str(uuid.uuid4())
            # Save attributes if provided, otherwise save "none"
            if attributes:
                attr_str = attributes
            else:
                attr_str = "none"
            
            conn = get_db()
            c = conn.cursor()
            c.execute('''INSERT INTO image_history (image_id, user_id, original_filename, generated_image, attributes)
                         VALUES (%s, %s, %s, %s, %s)''',
                      (image_id, current_user_id, file.filename, img_bytes, attr_str))
            conn.commit()
            conn.close()
        
        # Return format
        return_format = request.form.get('format', 'image')  # 'image' or 'base64'
        
        if return_format == 'base64':
            # Return as base64 JSON
            buffered = io.BytesIO()
            output_img.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            response = {
                'image': img_str, 
                'format': 'base64',
                'quality': quality,
                'enhancement': 'gfpgan' if use_gfpgan else 'pil',
                'size': output_img.size
            }
            if image_id:
                response['image_id'] = image_id
                response['saved'] = True
            return jsonify(response)
        else:
            # Return as image file
            img_io = io.BytesIO()
            output_img.save(img_io, 'PNG')
            img_io.seek(0)
            return send_file(img_io, mimetype='image/png')
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/generate-batch', methods=['POST'])
@token_required
def generate_batch(current_user_id):
    try:
        if 'images' not in request.files:
            return jsonify({'error': 'No images provided'}), 400
        
        files = request.files.getlist('images')
        attributes = request.form.get('attributes', None)
        quality = request.form.get('quality', 'medium')
        upscale = request.form.get('upscale', 'false').lower() == 'true'
        
        # Get enhancement method
        enhancement = request.form.get('enhancement', 'gfpgan')
        use_gfpgan = enhancement == 'gfpgan' and gfpgan_available
        
        if quality not in QUALITY_PRESETS:
            quality = 'medium'
        
        if attributes and attr_dim > 0:
            attr_list = [float(x) for x in attributes.split(',')]
            if len(attr_list) != attr_dim:
                return jsonify({'error': f'Attributes must be {attr_dim} values'}), 400
            attr_tensor = torch.tensor([attr_list], dtype=torch.float32).to(device)
        else:
            # No attributes or model doesn't use them - use zeros
            attr_tensor = torch.zeros(1, max(attr_dim, 1), dtype=torch.float32).to(device)
        
        results = []
        attr_str = ','.join(map(str, attr_list if attributes else [0.0, 0.0, 0.0, 0.0]))
        
        save_to_history = request.form.get('save', 'false').lower() == 'true'
        
        conn = get_db() if save_to_history else None
        c = conn.cursor() if conn else None
        
        for file in files:
            img = Image.open(file.stream).convert('RGB')
            
            # Preprocess input
            img = preprocess_input_image(img)
            
            sketch_tensor = transform(img).unsqueeze(0).to(device)
            
            # Create mask
            mask_np = create_mask_from_sketch(img)
            mask_tensor = torch.from_numpy(mask_np).float().unsqueeze(0).unsqueeze(0).to(device)
            
            with torch.no_grad():
                output = model(sketch_tensor, mask_tensor, attr_tensor)
            
            output = (output.squeeze().cpu() + 1) / 2
            output_img = T.ToPILImage()(output)
            
            # Apply post-processing enhancements
            output_img = enhance_output_image(output_img, quality, use_gfpgan=use_gfpgan)
            
            # Optionally upscale
            if upscale:
                new_size = (1024, 1024)
                output_img = output_img.resize(new_size, Image.Resampling.LANCZOS)
            
            buffered = io.BytesIO()
            output_img.save(buffered, format="PNG")
            img_bytes = buffered.getvalue()
            img_str = base64.b64encode(img_bytes).decode()
            
            result = {
                'filename': file.filename, 
                'image': img_str,
                'quality': quality,
                'enhancement': 'gfpgan' if use_gfpgan else 'pil',
                'size': output_img.size
            }
            
            if save_to_history:
                image_id = str(uuid.uuid4())
                c.execute('''INSERT INTO image_history (image_id, user_id, original_filename, generated_image, attributes)
                             VALUES (%s, %s, %s, %s, %s)''',
                          (image_id, current_user_id, file.filename, img_bytes, attr_str))
                result['image_id'] = image_id
                result['saved'] = True
            
            results.append(result)
        
        if conn:
            conn.commit()
            conn.close()
        
        return jsonify({'results': results, 'count': len(results)})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/history/save', methods=['POST'])
@token_required
def save_to_history(current_user_id):
    """
    Save an already-generated image to history without regenerating.
    Expects base64 encoded image in request body.
    """
    try:
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400
        
        # Get image data and metadata
        image_base64 = data['image']
        original_filename = data.get('filename', 'generated_image.png')
        attributes = data.get('attributes', 'none')
        
        # Decode base64 image
        try:
            image_bytes = base64.b64decode(image_base64)
        except Exception as e:
            return jsonify({'error': f'Invalid base64 image data: {str(e)}'}), 400
        
        # Generate unique image ID
        image_id = str(uuid.uuid4())
        
        # Save to database
        conn = get_db()
        c = conn.cursor()
        c.execute('''INSERT INTO image_history (image_id, user_id, original_filename, generated_image, attributes)
                     VALUES (%s, %s, %s, %s, %s)''',
                  (image_id, current_user_id, original_filename, image_bytes, attributes))
        conn.commit()
        conn.close()
        
        return jsonify({
            'message': 'Image saved to history successfully',
            'image_id': image_id,
            'saved': True
        }), 201
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False)
