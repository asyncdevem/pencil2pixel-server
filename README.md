# Pencil2Pixel Flask API

Flask REST API for sketch-to-image generation using conditional U-Net.

## Setup

```bash
pip install -r requirements.txt
```

## Run Server

```bash
python app.py
```

Server runs on `http://localhost:5000`

## API Endpoints

### 1. Health Check
```bash
GET /health
```

**Response:**
```json
{
  "status": "ok",
  "device": "cuda"
}
```

### 2. Generate Image (Single)
```bash
POST /generate
```

**Parameters:**
- `image` (file, required): Sketch image file
- `attributes` (string, optional): 4 comma-separated float values for tone controls (e.g., "0.25,0.75,0.33,0.33")
  - Values: skin_tone, hair_tone, eye_tone, lip_tone (each 0.0-1.0)
- `format` (string, optional): "image" (default) or "base64"
- `quality` (string, optional): "low", "medium" (default), "high", or "ultra"
- `upscale` (boolean, optional): "true" to upscale to 1024x1024, "false" (default)
- `save` (boolean, optional): "true" to save to history, "false" (default)

**Example (cURL - returns image):**
```bash
curl -X POST http://localhost:5000/generate \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -F "image=@sketch.jpg" \
  -F "quality=high" \
  -F "upscale=true" \
  --output result.png
```

**Example (cURL - with attributes):**
```bash
curl -X POST http://localhost:5000/generate \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -F "image=@sketch.jpg" \
  -F "attributes=0.25,0.75,0.33,0.33" \
  -F "format=base64"
```

**Example (Python):**
```python
import requests

# Login first
response = requests.post('http://localhost:5000/auth/login', json={
    'email': 'user@example.com',
    'password': 'password123'
})
token = response.json()['token']
headers = {'Authorization': f'Bearer {token}'}

# Generate image
with open('sketch.jpg', 'rb') as f:
    files = {'image': f}
    data = {
        'quality': 'high',
        'upscale': 'true',
        'format': 'image'
    }
    response = requests.post('http://localhost:5000/generate', 
                            files=files, data=data, headers=headers)
    
    with open('output.png', 'wb') as out:
        out.write(response.content)
```

### 3. Generate Batch
```bash
POST /generate-batch
```

**Parameters:**
- `images` (files, required): Multiple sketch image files
- `attributes` (string, optional): 4 comma-separated float values (applied to all images)
- `quality` (string, optional): "low", "medium" (default), "high", or "ultra"
- `upscale` (boolean, optional): "true" to upscale to 1024x1024
- `save` (boolean, optional): "true" to save to history

**Response:**
```json
{
  "results": [
    {
      "filename": "sketch1.jpg",
      "image": "base64_encoded_image..."
    }
  ],
  "count": 1
}
```

**Example (cURL):**
```bash
curl -X POST http://localhost:5000/generate-batch \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -F "images=@sketch1.jpg" \
  -F "images=@sketch2.jpg" \
  -F "quality=high"
```

## Authentication

The API uses JWT-based authentication. Most endpoints require a valid token.

### Signup
```bash
POST /auth/signup
Content-Type: application/json

{
  "username": "john_doe",
  "email": "john@example.com",
  "password": "password123"
}
```

### Login
```bash
POST /auth/login
Content-Type: application/json

{
  "email": "john@example.com",
  "password": "password123"
}
```

Returns a JWT token valid for 7 days. Include in subsequent requests:
```
Authorization: Bearer YOUR_TOKEN
```

## Attributes

The new model uses 4 floating-point tone attributes (optional):
- **skin_tone** (0.0-1.0): Skin tone control
- **hair_tone** (0.0-1.0): Hair tone control
- **eye_tone** (0.0-1.0): Eye tone control
- **lip_tone** (0.0-1.0): Lip tone control

If not provided, the model uses zero values and generates images based purely on the sketch input.

## Quality Presets

- **low**: Basic output (512x512)
- **medium**: Moderate sharpening and enhancement (default)
- **high**: Strong sharpening and color boost
- **ultra**: Maximum enhancement

Add `upscale=true` to get 1024x1024 output with high-quality resampling.

## Testing

Run the test script:
```bash
python test_api.py
```

Edit `test_api.py` to uncomment and test specific endpoints with your images.

## Mobile/App Integration

**For mobile apps (React Native, Flutter, etc.):**
- Use multipart/form-data for image upload
- Set `format=base64` to receive base64 string
- Include JWT token in Authorization header
- Decode base64 to display image

**Example (JavaScript/React Native):**
```javascript
const formData = new FormData();
formData.append('image', {
  uri: imageUri,
  type: 'image/jpeg',
  name: 'sketch.jpg'
});
formData.append('quality', 'high');
formData.append('format', 'base64');

fetch('http://localhost:5000/generate', {
  method: 'POST',
  headers: {
    'Authorization': `Bearer ${token}`
  },
  body: formData
})
.then(res => res.json())
.then(data => {
  const imageUri = `data:image/png;base64,${data.image}`;
  // Display imageUri in Image component
});
```

## Production Deployment

For production, use a WSGI server:
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

Or use Docker, AWS Lambda, Google Cloud Run, etc.
