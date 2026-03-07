#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for Weighbridge API with base64 support
"""

import requests
import base64
import json
import sys

def encode_image_to_base64(image_path):
    """Encode image file to base64"""
    with open(image_path, 'rb') as f:
        img_bytes = f.read()
    return base64.b64encode(img_bytes).decode('utf-8')

def save_base64_image(base64_string, output_path):
    """Save base64 string to image file"""
    img_bytes = base64.b64decode(base64_string)
    with open(output_path, 'wb') as f:
        f.write(img_bytes)

def test_api(image_path, weighbridge_id: 3, api_url='http://localhost:4001'):
    """Test the weighbridge validation API"""
    
    print(f"Testing API with:")
    print(f"  Image: {image_path}")
    print(f"  Weighbridge ID: {weighbridge_id}")
    print(f"  API URL: {api_url}")
    print("-" * 60)
    
    # 1. Test health endpoint
    print("\n1. Testing /health endpoint...")
    try:
        response = requests.get(f"{api_url}/health")
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.json()}")
    except Exception as e:
        print(f"   Error: {e}")
        return
    
    # 2. Encode image to base64
    print("\n2. Encoding image to base64...")
    try:
        image_base64 = encode_image_to_base64(image_path)
        print(f"   Base64 length: {len(image_base64)} characters")
    except Exception as e:
        print(f"   Error: {e}")
        return
    
    # 3. Test validation endpoint
    print("\n3. Testing /validate endpoint...")
    try:
        payload = {
            "weighbridge_id": weighbridge_id,
            "image_base64": image_base64
        }
        
        response = requests.post(
            f"{api_url}/validate",
            json=payload,
            headers={'Content-Type': 'application/json'}
        )
        
        print(f"   Status: {response.status_code}")
        
        result = response.json()
        print(f"\n   Results:")
        print(f"   - Valid: {result.get('valid', 'N/A')}")
        print(f"   - Description: {result.get('description', 'N/A')}")
        
        # Save processed image
        if 'processed_image_base64' in result and result['processed_image_base64']:
            output_path = 'test_output_annotated.jpg'
            save_base64_image(result['processed_image_base64'], output_path)
            print(f"   - Processed image saved to: {output_path}")
        else:
            print(f"   - No processed image returned")
            
    except Exception as e:
        print(f"   Error: {e}")
        return
    
    print("\n" + "=" * 60)
    print("Test completed successfully!")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python test_api.py <image_path> [weighbridge_id] [api_url]")
        print("\nExample:")
        print("  python test_api.py baskol-test/bask2_up_20251202_110500.jpg 2")
        sys.exit(1)
    
    image_path = sys.argv[1]
    weighbridge_id: 3
    api_url = sys.argv[3] if len(sys.argv) > 3 else 'http://localhost:4001'
    
    test_api(image_path, weighbridge_id, api_url)
