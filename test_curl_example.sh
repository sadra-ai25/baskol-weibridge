#!/bin/bash

# مثال تست API با curl

echo "=============================================="
echo "  Weighbridge API Test with cURL"
echo "=============================================="
echo ""

# Check if image path is provided
if [ -z "$1" ]; then
    echo "Usage: ./test_curl_example.sh <image_path> [weighbridge_id]"
    echo ""
    echo "Example:"
    echo "  ./test_curl_example.sh baskol-test/bask2_up_20251206_111619.jpg 3"
    exit 1
fi

IMAGE_PATH=$1
WEIGHBRIDGE_ID=${2:-3}
API_URL="http://localhost:4001"

echo "Image: $IMAGE_PATH"
echo "Weighbridge ID: $WEIGHBRIDGE_ID"
echo "API URL: $API_URL"
echo ""

# 1. Test health
echo "1. Testing /health endpoint..."
curl -s "$API_URL/health" | python3 -m json.tool
echo ""

# 2. Encode image to base64
echo "2. Encoding image to base64..."
IMAGE_BASE64=$(base64 -w 0 "$IMAGE_PATH")
echo "   Base64 length: ${#IMAGE_BASE64} characters"
echo ""

# 3. Send validation request
echo "3. Sending validation request..."
RESPONSE=$(curl -s -X POST "$API_URL/validate" \
  -H "Content-Type: application/json" \
  -d "{
    \"weighbridge_id\": $WEIGHBRIDGE_ID,
    \"image_base64\": \"$IMAGE_BASE64\"
  }")

# Parse response
VALID=$(echo "$RESPONSE" | python3 -c "import sys, json; print(json.load(sys.stdin).get('valid', 'N/A'))")
DESCRIPTION=$(echo "$RESPONSE" | python3 -c "import sys, json; print(json.load(sys.stdin).get('description', 'N/A'))")
PROCESSED_IMAGE=$(echo "$RESPONSE" | python3 -c "import sys, json; print(json.load(sys.stdin).get('processed_image_base64', ''))")

echo ""
echo "Results:"
echo "  Valid: $VALID"
echo "  Description: $DESCRIPTION"
echo ""

# 4. Save processed image
if [ -n "$PROCESSED_IMAGE" ]; then
    OUTPUT_FILE="test_output_$(date +%Y%m%d_%H%M%S).jpg"
    echo "$PROCESSED_IMAGE" | base64 -d > "$OUTPUT_FILE"
    echo "  Processed image saved to: $OUTPUT_FILE"
else
    echo "  No processed image returned"
fi

echo ""
echo "=============================================="
echo "  Test completed!"
echo "=============================================="
