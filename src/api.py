#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Weighbridge Validation API
Single image validation endpoint with base64 support
"""

from flask import Flask, request, jsonify
import os
import json
import cv2
import tempfile
import base64
import numpy as np
from simple_validator import SimpleValidator

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32MB max for base64

# Load validator
validator = SimpleValidator(model_path='weights/best.pt', confidence_threshold=0.25)

# Weighbridge ID mapping: software ID -> config ID
# فقط باسکول 1 و 3 وجود دارد
WEIGHBRIDGE_MAPPING = {
    1: 1,  # Weighbridge 1 -> Config wb1
    3: 3   # Weighbridge 3 -> Config wb3
}

def decode_base64_image(base64_string):
    """Decode base64 string to image"""
    try:
        # Remove header if present (data:image/jpeg;base64,)
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]

        # Decode base64 to bytes
        img_bytes = base64.b64decode(base64_string)

        # Convert to numpy array
        nparr = np.frombuffer(img_bytes, np.uint8)

        # Decode image
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise ValueError("Failed to decode image")

        return img
    except Exception as e:
        raise ValueError(f"Invalid base64 image: {str(e)}")

def encode_image_to_base64(image_path):
    """Encode image file to base64 string"""
    try:
        with open(image_path, 'rb') as f:
            img_bytes = f.read()

        base64_string = base64.b64encode(img_bytes).decode('utf-8')
        return base64_string
    except Exception as e:
        raise ValueError(f"Failed to encode image: {str(e)}")

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'weighbridge-validation-api',
        'version': '2.0.0'
    })

@app.route('/validate', methods=['POST'])
def validate_image():
    """
    Validate weighbridge image with base64 support

    Request JSON:
        {
            "weighbridge_id": 1,  // 1 or 3 (maps to config wb1 or wb3)
            "image_base64": "base64_encoded_image_string"
        }

    Response JSON:
        {
            "valid": true,
            "description": "VALID - Vehicle properly positioned in yellow area",
            "processed_image_base64": "base64_encoded_annotated_image"
        }
    """
    try:
        # Parse JSON request
        data = request.get_json()

        if not data:
            return jsonify({
                'valid': False,
                'description': 'Invalid JSON request',
                'processed_image_base64': ''
            }), 400

        # Validate required fields
        if 'weighbridge_id' not in data:
            return jsonify({
                'valid': False,
                'description': 'Missing weighbridge_id parameter',
                'processed_image_base64': ''
            }), 400

        if 'image_base64' not in data:
            return jsonify({
                'valid': False,
                'description': 'Missing image_base64 parameter',
                'processed_image_base64': ''
            }), 400

        # Get parameters
        try:
            software_id = int(data['weighbridge_id'])
            if software_id not in WEIGHBRIDGE_MAPPING:
                raise ValueError(f"Invalid weighbridge_id. Must be 1 or 3")
        except ValueError as e:
            return jsonify({
                'valid': False,
                'description': f'Invalid weighbridge_id: {str(e)}',
                'processed_image_base64': ''
            }), 400

        # Map software ID to config ID
        config_id = WEIGHBRIDGE_MAPPING[software_id]

        # Load config
        config_path = f'configs/wb{config_id}_single_cam_roi.json'
        if not os.path.exists(config_path):
            return jsonify({
                'valid': False,
                'description': f'Config file not found for weighbridge {software_id}',
                'processed_image_base64': ''
            }), 404

        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        # Decode base64 image
        try:
            img = decode_base64_image(data['image_base64'])
        except ValueError as e:
            return jsonify({
                'valid': False,
                'description': f'Invalid base64 image: {str(e)}',
                'processed_image_base64': ''
            }), 400

        # Save image to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_input:
            cv2.imwrite(temp_input.name, img)
            input_path = temp_input.name

        try:
            # Validate image
            result = validator.validate(input_path, config)

            # Generate annotated image
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_output:
                output_path = temp_output.name

            validator.draw_results(input_path, config, result, output_path)

            # Encode annotated image to base64
            processed_image_base64 = encode_image_to_base64(output_path)

            # Prepare response
            response_data = {
                'valid': result.is_valid,
                'description': result.message,
                'processed_image_base64': processed_image_base64
            }

            # Clean up temp files
            if os.path.exists(output_path):
                os.unlink(output_path)

            return jsonify(response_data)

        finally:
            # Clean up input temp file
            if os.path.exists(input_path):
                os.unlink(input_path)

    except Exception as e:
        return jsonify({
            'valid': False,
            'description': f'Internal server error: {str(e)}',
            'processed_image_base64': ''
        }), 500

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs('outputs', exist_ok=True)

    # Run server
    app.run(host='0.0.0.0', port=4001, debug=False)
